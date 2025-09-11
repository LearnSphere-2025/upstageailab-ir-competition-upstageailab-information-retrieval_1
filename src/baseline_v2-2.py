#!/usr/bin/env python3
"""
Information Retrieval Baseline System v2

한국어 과학 상식 질문답변을 위한 RAG (Retrieval Augmented Generation) 시스템

주요 기능:
- Elasticsearch를 활용한 하이브리드 검색 (BM25 + 벡터 유사도)
- KLUE RoBERTa 임베딩 모델로 한국어 의미 검색
- OpenAI GPT-3.5를 통한 질문답변 생성
- 220개 과학 상식 평가 데이터셋 지원

주의: 코드에서는 dense_retrieve 메서드가 정의되어 있지만, 실제 RAG 파이프라인(answer_question)에서는 희소 검색(sparse_retrieve)만 사용하고 있습니다 (331번째 줄). 하이브리드 검색이라고 하려면 두 방식을 결합해야 하는데, 현재는 BM25만 실제로 활용되고 있는 상태입니다
"""

import os
import json
import time
import traceback
from pathlib import Path
from subprocess import Popen, PIPE, STDOUT
from typing import List, Dict, Any, Optional

from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv


class Config:
    """IR 시스템 설정 관리 클래스"""
    
    def __init__(self):
        load_dotenv('./config/.env')
        self._load_env_config()
        self._validate()
    
    def _load_env_config(self):
        """환경변수에서 설정 로드"""
        # 모델 설정
        self.embedding_model = os.getenv('EMBEDDING_MODEL', 'klue/roberta-large')
        self.llm_model = os.getenv('LLM_MODEL', 'gpt-3.5-turbo-1106')
        
        # Elasticsearch 설정
        self.es_host = os.getenv('ES_HOST', 'localhost')
        self.es_port = os.getenv('ES_PORT', '9200')
        self.es_index_name = os.getenv('ES_INDEX_NAME', 'ir_documents_klue')
        self.es_cert_path = os.getenv('ES_CERT_PATH', './elasticsearch-8.8.0/config/certs/http_ca.crt')
        self.es_binary_path = os.getenv('ES_BINARY_PATH', './elasticsearch-8.8.0/bin/elasticsearch')
        
        # 인증 정보
        self.es_username = os.getenv('ES_USERNAME', 'elastic')
        self.es_password = os.getenv('ES_PASSWORD')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        
        # 데이터 경로
        self.documents_path = os.getenv('DOCUMENTS_PATH', './data/documents.jsonl')
        self.eval_path = os.getenv('EVAL_PATH', './data/eval.jsonl')
        
        # 실험 설정
        self.experiment_name = os.getenv('EXPERIMENT_NAME', 'klue_roberta')
        self.results_path = os.getenv('RESULTS_PATH', f'./results/submission_{self.experiment_name}.csv')
        
        print(f"✓ Loaded config from environment variables")
        print(f"  Experiment: {self.experiment_name}")
        print(f"  Embedding model: {self.embedding_model}")
        print(f"  ES Index: {self.es_index_name}")
    
    def _validate(self):
        """필수 설정 검증"""
        if not self.es_password:
            raise ValueError("ES_PASSWORD environment variable is required")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
    
    def __str__(self) -> str:
        """설정 정보 출력 (민감 정보 제외)"""
        return f"""Configuration - {self.experiment_name}:
  Elasticsearch: {self.es_host}:{self.es_port}
  Index: {self.es_index_name}
  Documents: {self.documents_path}
  Evaluation: {self.eval_path}
  LLM Model: {self.llm_model}
  Embedding Model: {self.embedding_model}
  Results: {self.results_path}"""


class IRBaselineSystem:
    """정보검색 기반 RAG 시스템"""
    
    def __init__(self, config_obj: Optional[Config] = None):
        self.config = config_obj or Config()
        self.es: Optional[Elasticsearch] = None
        self.model: Optional[SentenceTransformer] = None
        self.client: Optional[OpenAI] = None
        self.es_server: Optional[Popen] = None
        
    def setup_elasticsearch(self, start_server: bool = True):
        """Elasticsearch 서버 시작 및 클라이언트 설정"""
        if start_server:
            print("Starting Elasticsearch server...")
            
            es_binary = Path(self.config.es_binary_path)
            if not es_binary.exists():
                raise FileNotFoundError(f"Elasticsearch binary not found at {self.config.es_binary_path}")
            
            try:
                self.es_server = Popen([str(es_binary)], stdout=PIPE, stderr=STDOUT)
                time.sleep(30)  # 서버 시작 대기
            except OSError as e:
                print(f"Warning: Could not start Elasticsearch: {e}")
                print("Assuming Elasticsearch is already running...")
        
        # 클라이언트 생성
        es_url = f"https://{self.config.es_host}:{self.config.es_port}"
        cert_path = Path(self.config.es_cert_path)
        ca_certs = str(cert_path) if cert_path.exists() else False
        
        self.es = Elasticsearch(
            [es_url], 
            basic_auth=(self.config.es_username, self.config.es_password), 
            ca_certs=ca_certs,
            verify_certs=ca_certs is not False
        )
        
        try:
            info = self.es.info()
            print("Elasticsearch client info:", info)
        except Exception as e:
            print(f"Warning: Could not connect to Elasticsearch: {e}")
            raise
        
    def setup_models(self):
        """임베딩 모델 및 OpenAI 클라이언트 초기화"""
        print(f"Loading Sentence Transformer model: {self.config.embedding_model}")
        try:
            self.model = SentenceTransformer(self.config.embedding_model)
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            raise
        
        print("Setting up OpenAI client...")
        try:
            self.client = OpenAI(api_key=self.config.openai_api_key)
        except Exception as e:
            print(f"Error setting up OpenAI client: {e}")
            raise
        
    def get_embedding(self, sentences: List[str]) -> List[List[float]]:
        """문장들의 임베딩 생성"""
        return self.model.encode(sentences)
    
    def get_embeddings_in_batches(self, docs: List[Dict], batch_size: int = 100) -> List[List[float]]:
        """배치 단위로 문서 임베딩 생성"""
        batch_embeddings = []
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i + batch_size]
            contents = [doc["content"] for doc in batch]
            embeddings = self.get_embedding(contents)
            batch_embeddings.extend(embeddings)
            print(f'Processed batch {i // batch_size + 1}/{(len(docs) - 1) // batch_size + 1}')
        return batch_embeddings
    
    def setup_index(self):
        """한국어 분석기와 임베딩을 지원하는 Elasticsearch 인덱스 설정"""
        settings = {
            "analysis": {
                "analyzer": {
                    "nori": {
                        "type": "custom",
                        "tokenizer": "nori_tokenizer",
                        "decompound_mode": "mixed",
                        "filter": ["nori_posfilter"]
                    }
                },
                "filter": {
                    "nori_posfilter": {
                        "type": "nori_part_of_speech",
                        "stoptags": ["E", "J", "SC", "SE", "SF", "VCN", "VCP", "VX"]
                    }
                }
            }
        }
        
        mappings = {
            "properties": {
                "content": {"type": "text", "analyzer": "nori"},
                "embeddings": {
                    "type": "dense_vector",
                    "dims": 1024,
                    "index": True,
                    "similarity": "l2_norm"
                }
            }
        }
        
        if self.es.indices.exists(index=self.config.es_index_name):
            self.es.indices.delete(index=self.config.es_index_name)
        
        self.es.indices.create(
            index=self.config.es_index_name, 
            settings=settings, 
            mappings=mappings
        )
        
    def index_documents(self):
        """문서들을 임베딩과 함께 인덱싱"""
        print("Loading and indexing documents...")
        
        docs_path = Path(self.config.documents_path)
        if not docs_path.exists():
            raise FileNotFoundError(f"Documents file not found at {self.config.documents_path}")
        
        with open(docs_path) as f:
            docs = [json.loads(line) for line in f]
        
        print(f"Loaded {len(docs)} documents")
        embeddings = self.get_embeddings_in_batches(docs)
        
        index_docs = []
        for doc, embedding in zip(docs, embeddings):
            doc["embeddings"] = embedding.tolist()
            index_docs.append(doc)
        
        actions = [
            {"_index": self.config.es_index_name, "_source": doc}
            for doc in index_docs
        ]
        
        result = helpers.bulk(self.es, actions)
        print(f"Indexed documents: {result}")
        
    def sparse_retrieve(self, query_str: str, size: int = 10) -> Dict:
        """BM25를 사용한 희소 검색"""
        query = {
            "match": {
                "content": {"query": query_str}
            }
        }
        return self.es.search(
            index=self.config.es_index_name, 
            query=query, 
            size=size, 
            sort="_score"
        )
    
    def dense_retrieve(self, query_str: str, size: int = 10) -> Dict:
        """벡터 유사도를 사용한 밀집 검색"""
        query_embedding = self.get_embedding([query_str])[0]
        
        knn = {
            "field": "embeddings",
            "query_vector": query_embedding.tolist(),
            "k": size,
            "num_candidates": 100
        }
        
        return self.es.search(index=self.config.es_index_name, knn=knn)
    
    def answer_question(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """RAG를 사용한 질문 답변"""
        # 검색 함수 호출을 위한 시스템 프롬프트
        function_calling_prompt = """
## Role: 과학 상식 전문가

## Instruction
- 사용자가 과학 지식에 관한 질문을 하면 search API를 호출하여 관련 정보를 검색합니다.
- 과학 상식과 관련되지 않은 질문에는 직접 답변합니다.
"""
        
        # QA 생성을 위한 시스템 프롬프트
        qa_prompt = """
## Role: 과학 상식 전문가

## Instructions
- 사용자의 질문과 제공된 참고 문서를 바탕으로 정확하고 간결한 답변을 생성합니다.
- 제공된 정보로 답변할 수 없는 경우, 정보 부족을 명시합니다.
- 한국어로 답변합니다.
"""
        
        # 검색 함수 정의
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "관련 문서 검색",
                    "parameters": {
                        "properties": {
                            "standalone_query": {
                                "type": "string",
                                "description": "검색에 사용할 독립적인 쿼리"
                            }
                        },
                        "required": ["standalone_query"],
                        "type": "object"
                    }
                }
            },
        ]
        
        response = {
            "standalone_query": "", 
            "topk": [], 
            "references": [], 
            "answer": ""
        }
        
        # 함수 호출 단계
        function_messages = [{"role": "system", "content": function_calling_prompt}] + messages
        
        try:
            result = self.client.chat.completions.create(
                model=self.config.llm_model,
                messages=function_messages,
                tools=tools,
                temperature=0,
                seed=1,
                timeout=10
            )
        except Exception as e:
            print(f"Error in function calling: {e}")
            traceback.print_exc()
            return response
        
        # 검색이 필요한 경우
        if result.choices[0].message.tool_calls:
            tool_call = result.choices[0].message.tool_calls[0]
            function_args = json.loads(tool_call.function.arguments)
            standalone_query = function_args.get("standalone_query", "")
            
            # 희소 검색 수행
            search_result = self.sparse_retrieve(standalone_query, 3)
            
            response["standalone_query"] = standalone_query
            retrieved_context = []
            
            for rst in search_result['hits']['hits']:
                content = rst["_source"]["content"]
                retrieved_context.append(content)
                response["topk"].append(rst["_source"]["docid"])
                response["references"].append({
                    "score": rst["_score"], 
                    "content": content
                })
            
            # QA 생성 단계
            context_message = {"role": "assistant", "content": json.dumps(retrieved_context)}
            qa_messages = [{"role": "system", "content": qa_prompt}] + messages + [context_message]
            
            try:
                qa_result = self.client.chat.completions.create(
                    model=self.config.llm_model,
                    messages=qa_messages,
                    temperature=0,
                    seed=1,
                    timeout=30
                )
                response["answer"] = qa_result.choices[0].message.content
            except Exception as e:
                print(f"Error in QA generation: {e}")
                traceback.print_exc()
                
        else:
            # 직접 답변
            response["answer"] = result.choices[0].message.content
        
        return response
    
    def eval_rag(self, eval_filename: str, output_filename: str, limit: Optional[int] = None):
        """평가 데이터셋에 대한 RAG 성능 평가"""
        eval_path = Path(eval_filename)
        if not eval_path.exists():
            raise FileNotFoundError(f"Evaluation file not found at {eval_filename}")
        
        output_path = Path(output_filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(eval_path) as f, open(output_filename, "w") as of:
            for idx, line in enumerate(f):
                if limit and idx >= limit:
                    break
                    
                data = json.loads(line)
                print(f'Test {idx}\nQuestion: {data["msg"]}')
                
                response = self.answer_question(data["msg"])
                print(f'Answer: {response["answer"]}\n')
                
                output = {
                    "eval_id": data["eval_id"], 
                    "standalone_query": response["standalone_query"], 
                    "topk": response["topk"], 
                    "answer": response["answer"], 
                    "references": response["references"]
                }
                
                of.write(f'{json.dumps(output, ensure_ascii=False)}\n')
    
    def test_retrieval(self, query: str = "금성이 다른 행성들보다 밝게 보이는 이유는 무엇인가요?"):
        """검색 기능 테스트"""
        print(f"Testing with query: {query}")
        
        print("\n=== Sparse Retrieval Results ===")
        sparse_results = self.sparse_retrieve(query, 3)
        for rst in sparse_results['hits']['hits']:
            print(f'Score: {rst["_score"]:.5f}')
            print(f'Content: {rst["_source"]["content"][:200]}...\n')
        
        print("\n=== Dense Retrieval Results ===")
        dense_results = self.dense_retrieve(query, 3)
        for rst in dense_results['hits']['hits']:
            print(f'Score: {rst["_score"]:.7f}')
            print(f'Content: {rst["_source"]["content"][:200]}...\n')


def main(start_es_server: bool = True, 
         run_indexing: bool = True, 
         run_evaluation: bool = True, 
         eval_limit: Optional[int] = None) -> int:
    """IR 시스템 메인 실행 함수"""
    config = Config()
    print("Starting IR Baseline System...")
    print(config)
    
    system = IRBaselineSystem(config)
    
    try:
        # 시스템 구성 요소 설정
        system.setup_elasticsearch(start_server=start_es_server)
        system.setup_models()
        
        # 인덱싱 실행
        if run_indexing:
            system.setup_index()
            system.index_documents()
        
        # 검색 테스트
        if Path(config.documents_path).exists():
            try:
                system.test_retrieval()
            except Exception as e:
                print(f"Warning: Could not test retrieval: {e}")
        else:
            print("Warning: Documents file not found, skipping retrieval test")
        
        # 평가 실행
        if run_evaluation and Path(config.eval_path).exists():
            print("\nRunning evaluation...")
            system.eval_rag(config.eval_path, config.results_path, limit=eval_limit)
            print("Evaluation completed!")
        elif run_evaluation:
            print("Warning: Evaluation file not found, skipping evaluation")
            
    except Exception as e:
        print(f"Error during execution: {e}")
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='IR Baseline System v2')
    parser.add_argument('--no-es-server', action='store_true', 
                        help='Elasticsearch 서버를 시작하지 않음 (이미 실행 중인 경우)')
    parser.add_argument('--no-indexing', action='store_true',
                        help='문서 인덱싱 건너뛰기')
    parser.add_argument('--no-evaluation', action='store_true',
                        help='평가 건너뛰기')
    parser.add_argument('--eval-limit', type=int, default=None,
                        help='평가 샘플 수 제한')
    
    args = parser.parse_args()
    
    exit_code = main(
        start_es_server=not args.no_es_server,
        run_indexing=not args.no_indexing,
        run_evaluation=not args.no_evaluation,
        eval_limit=args.eval_limit
    )
    
    exit(exit_code)