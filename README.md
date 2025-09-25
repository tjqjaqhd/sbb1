# Todo Backend API

Node.js + Express 기반의 Todo 앱 백엔드 API입니다.

## 🚀 기능

- ✅ Todo 목록 조회 (GET)
- ✅ 새 Todo 생성 (POST)
- ✅ Todo 완료/미완료 토글 (PUT)
- ✅ Todo 삭제 (DELETE)
- ✅ Todo 통계 조회
- ✅ CORS 지원
- ✅ 포괄적인 에러 핸들링
- ✅ 입력 데이터 검증
- ✅ 정적 파일 서빙

## 🛠️ 설치 및 실행

### 1. 의존성 설치
```bash
npm install
```

### 2. 서버 시작
```bash
# 프로덕션 모드
npm start

# 개발 모드 (nodemon)
npm run dev
```

### 3. 서버 접속
- 웹 브라우저에서 http://localhost:3000 접속
- API 베이스 URL: http://localhost:3000/api

## 📡 API 엔드포인트

### 1. Todo 목록 조회
```http
GET /api/todos
```

**응답 예시:**
```json
{
  "success": true,
  "data": [
    {
      "id": 1,
      "text": "React 공부하기",
      "completed": false,
      "createdAt": "2025-01-15T12:00:00.000Z"
    }
  ],
  "total": 1,
  "message": "Todo 목록을 성공적으로 조회했습니다."
}
```

### 2. 새 Todo 생성
```http
POST /api/todos
Content-Type: application/json

{
  "text": "새로운 할 일"
}
```

**응답 예시:**
```json
{
  "success": true,
  "data": {
    "id": 2,
    "text": "새로운 할 일",
    "completed": false,
    "createdAt": "2025-01-15T12:05:00.000Z"
  },
  "message": "Todo가 성공적으로 생성되었습니다."
}
```

### 3. Todo 완료/미완료 토글
```http
PUT /api/todos/1
```

**응답 예시:**
```json
{
  "success": true,
  "data": {
    "id": 1,
    "text": "React 공부하기",
    "completed": true,
    "createdAt": "2025-01-15T12:00:00.000Z"
  },
  "message": "Todo가 성공적으로 완료로 변경되었습니다."
}
```

### 4. Todo 삭제
```http
DELETE /api/todos/1
```

**응답 예시:**
```json
{
  "success": true,
  "data": {
    "id": 1,
    "text": "React 공부하기",
    "completed": true,
    "createdAt": "2025-01-15T12:00:00.000Z"
  },
  "message": "Todo가 성공적으로 삭제되었습니다."
}
```

### 5. Todo 통계 조회
```http
GET /api/todos/stats
```

**응답 예시:**
```json
{
  "success": true,
  "data": {
    "total": 5,
    "completed": 3,
    "pending": 2,
    "completionRate": 60
  },
  "message": "Todo 통계 정보를 성공적으로 조회했습니다."
}
```

## 📋 데이터 구조

### Todo 객체
```typescript
interface Todo {
  id: number;          // 고유 식별자
  text: string;        // 할 일 내용 (1-500자)
  completed: boolean;  // 완료 상태
  createdAt: string;   // 생성 시간 (ISO 8601)
}
```

### API 응답 구조
```typescript
interface ApiResponse<T> {
  success: boolean;    // 성공 여부
  data?: T;           // 응답 데이터
  message: string;    // 메시지
  total?: number;     // 총 개수 (목록 조회 시)
  error?: string;     // 에러 코드
  details?: string[]; // 상세 오류 정보
}
```

## ⚠️ 에러 처리

API는 다음과 같은 에러 상황을 처리합니다:

- **400 Bad Request**: 잘못된 요청 데이터
- **404 Not Found**: 존재하지 않는 Todo ID
- **500 Internal Server Error**: 서버 내부 오류

**에러 응답 예시:**
```json
{
  "success": false,
  "error": "VALIDATION_ERROR",
  "message": "입력 데이터가 유효하지 않습니다.",
  "details": [
    "text는 필수 입력값이며 문자열이어야 합니다."
  ]
}
```

## 🔧 기술 스택

- **Node.js**: 런타임 환경
- **Express.js**: 웹 프레임워크
- **CORS**: Cross-Origin Resource Sharing 지원
- **메모리 저장소**: 간단한 배열 기반 데이터 저장

## 📁 프로젝트 구조

```
sbb1/
├── server.js          # 메인 서버 파일
├── package.json       # 프로젝트 설정
├── README.md          # 프로젝트 문서
└── public/           # 정적 파일
    └── index.html     # API 테스트 페이지
```

## 🧪 API 테스트

### curl 사용 예시
```bash
# Todo 목록 조회
curl -X GET http://localhost:3000/api/todos

# 새 Todo 생성
curl -X POST http://localhost:3000/api/todos \
  -H "Content-Type: application/json" \
  -d '{"text": "Node.js 공부하기"}'

# Todo 완료 토글
curl -X PUT http://localhost:3000/api/todos/1

# Todo 삭제
curl -X DELETE http://localhost:3000/api/todos/1
```

### JavaScript fetch 사용 예시
```javascript
// Todo 목록 조회
const todos = await fetch('/api/todos').then(res => res.json());

// 새 Todo 생성
const newTodo = await fetch('/api/todos', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ text: '새로운 할 일' })
}).then(res => res.json());

// Todo 완료 토글
const updatedTodo = await fetch('/api/todos/1', {
  method: 'PUT'
}).then(res => res.json());

// Todo 삭제
const deletedTodo = await fetch('/api/todos/1', {
  method: 'DELETE'
}).then(res => res.json());
```

## 🔒 보안 고려사항

- 입력 데이터 검증 및 살균
- JSON 파싱 에러 처리
- 적절한 HTTP 상태 코드 사용
- 에러 정보 노출 제한 (프로덕션 환경)

## 📝 라이선스

ISC License

## 👨‍💻 개발자

Claude Code Assistant