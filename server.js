const express = require('express');
const cors = require('cors');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3000;

// 메모리 기반 Todo 데이터 저장소
let todos = [];
let nextId = 1;

// 미들웨어 설정
app.use(cors()); // CORS 활성화 (프론트엔드 연동을 위해)
app.use(express.json()); // JSON 파싱
app.use(express.static(path.join(__dirname, 'public'))); // 정적 파일 서빙

// 입력 검증 함수
const validateTodoData = (data) => {
  const errors = [];

  if (!data.text || typeof data.text !== 'string') {
    errors.push('text는 필수 입력값이며 문자열이어야 합니다.');
  }

  if (data.text && data.text.trim().length === 0) {
    errors.push('text는 빈 문자열일 수 없습니다.');
  }

  if (data.text && data.text.length > 500) {
    errors.push('text는 500자를 초과할 수 없습니다.');
  }

  return errors;
};

// ID 검증 함수
const validateId = (id) => {
  const numId = parseInt(id);
  return !isNaN(numId) && numId > 0;
};

// Todo 검색 함수
const findTodoById = (id) => {
  const numId = parseInt(id);
  return todos.find(todo => todo.id === numId);
};

// API 엔드포인트

// 1. GET /api/todos - 전체 Todo 목록 조회
app.get('/api/todos', (req, res) => {
  try {
    // 생성일자 기준 내림차순 정렬 (최신순)
    const sortedTodos = [...todos].sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt));

    res.status(200).json({
      success: true,
      data: sortedTodos,
      total: sortedTodos.length,
      message: 'Todo 목록을 성공적으로 조회했습니다.'
    });
  } catch (error) {
    console.error('Todo 목록 조회 중 오류:', error);
    res.status(500).json({
      success: false,
      error: 'INTERNAL_SERVER_ERROR',
      message: '서버 내부 오류가 발생했습니다.',
      details: process.env.NODE_ENV === 'development' ? error.message : undefined
    });
  }
});

// 2. POST /api/todos - 새 Todo 추가
app.post('/api/todos', (req, res) => {
  try {
    const validationErrors = validateTodoData(req.body);

    if (validationErrors.length > 0) {
      return res.status(400).json({
        success: false,
        error: 'VALIDATION_ERROR',
        message: '입력 데이터가 유효하지 않습니다.',
        details: validationErrors
      });
    }

    const newTodo = {
      id: nextId++,
      text: req.body.text.trim(),
      completed: false,
      createdAt: new Date().toISOString()
    };

    todos.push(newTodo);

    res.status(201).json({
      success: true,
      data: newTodo,
      message: 'Todo가 성공적으로 생성되었습니다.'
    });
  } catch (error) {
    console.error('Todo 생성 중 오류:', error);
    res.status(500).json({
      success: false,
      error: 'INTERNAL_SERVER_ERROR',
      message: '서버 내부 오류가 발생했습니다.',
      details: process.env.NODE_ENV === 'development' ? error.message : undefined
    });
  }
});

// 3. PUT /api/todos/:id - Todo 완료/미완료 토글
app.put('/api/todos/:id', (req, res) => {
  try {
    const { id } = req.params;

    // ID 유효성 검증
    if (!validateId(id)) {
      return res.status(400).json({
        success: false,
        error: 'INVALID_ID',
        message: 'ID는 양의 정수여야 합니다.',
        details: [`제공된 ID: ${id}`]
      });
    }

    const todo = findTodoById(id);

    if (!todo) {
      return res.status(404).json({
        success: false,
        error: 'TODO_NOT_FOUND',
        message: '해당 ID의 Todo를 찾을 수 없습니다.',
        details: [`요청된 ID: ${id}`]
      });
    }

    // completed 상태 토글
    todo.completed = !todo.completed;

    res.status(200).json({
      success: true,
      data: todo,
      message: `Todo가 성공적으로 ${todo.completed ? '완료' : '미완료'}로 변경되었습니다.`
    });
  } catch (error) {
    console.error('Todo 업데이트 중 오류:', error);
    res.status(500).json({
      success: false,
      error: 'INTERNAL_SERVER_ERROR',
      message: '서버 내부 오류가 발생했습니다.',
      details: process.env.NODE_ENV === 'development' ? error.message : undefined
    });
  }
});

// 4. DELETE /api/todos/:id - Todo 삭제
app.delete('/api/todos/:id', (req, res) => {
  try {
    const { id } = req.params;

    // ID 유효성 검증
    if (!validateId(id)) {
      return res.status(400).json({
        success: false,
        error: 'INVALID_ID',
        message: 'ID는 양의 정수여야 합니다.',
        details: [`제공된 ID: ${id}`]
      });
    }

    const todoIndex = todos.findIndex(todo => todo.id === parseInt(id));

    if (todoIndex === -1) {
      return res.status(404).json({
        success: false,
        error: 'TODO_NOT_FOUND',
        message: '해당 ID의 Todo를 찾을 수 없습니다.',
        details: [`요청된 ID: ${id}`]
      });
    }

    const deletedTodo = todos.splice(todoIndex, 1)[0];

    res.status(200).json({
      success: true,
      data: deletedTodo,
      message: 'Todo가 성공적으로 삭제되었습니다.'
    });
  } catch (error) {
    console.error('Todo 삭제 중 오류:', error);
    res.status(500).json({
      success: false,
      error: 'INTERNAL_SERVER_ERROR',
      message: '서버 내부 오류가 발생했습니다.',
      details: process.env.NODE_ENV === 'development' ? error.message : undefined
    });
  }
});

// 기타 엔드포인트

// 5. GET /api/todos/stats - Todo 통계 정보
app.get('/api/todos/stats', (req, res) => {
  try {
    const total = todos.length;
    const completed = todos.filter(todo => todo.completed).length;
    const pending = total - completed;

    res.status(200).json({
      success: true,
      data: {
        total,
        completed,
        pending,
        completionRate: total > 0 ? Math.round((completed / total) * 100) : 0
      },
      message: 'Todo 통계 정보를 성공적으로 조회했습니다.'
    });
  } catch (error) {
    console.error('Todo 통계 조회 중 오류:', error);
    res.status(500).json({
      success: false,
      error: 'INTERNAL_SERVER_ERROR',
      message: '서버 내부 오류가 발생했습니다.'
    });
  }
});

// 루트 경로 - API 정보 제공
app.get('/', (req, res) => {
  res.json({
    name: 'Todo Backend API',
    version: '1.0.0',
    description: 'Node.js + Express 기반 Todo 앱 백엔드 API',
    endpoints: {
      'GET /api/todos': 'Todo 목록 조회',
      'POST /api/todos': 'Todo 생성',
      'PUT /api/todos/:id': 'Todo 완료/미완료 토글',
      'DELETE /api/todos/:id': 'Todo 삭제',
      'GET /api/todos/stats': 'Todo 통계 조회'
    },
    author: 'Claude Code Assistant',
    timestamp: new Date().toISOString()
  });
});

// API 경로가 존재하지 않는 경우 처리
app.use('/api/*', (req, res) => {
  res.status(404).json({
    success: false,
    error: 'API_NOT_FOUND',
    message: '요청한 API 엔드포인트를 찾을 수 없습니다.',
    details: [`요청 경로: ${req.originalUrl}`, `허용된 방법: ${req.method}`]
  });
});

// 전역 에러 핸들러
app.use((error, req, res, next) => {
  console.error('예상치 못한 오류 발생:', error);

  // JSON 파싱 에러 처리
  if (error instanceof SyntaxError && error.status === 400 && 'body' in error) {
    return res.status(400).json({
      success: false,
      error: 'INVALID_JSON',
      message: '잘못된 JSON 형식입니다.',
      details: ['요청 본문의 JSON 형식을 확인해주세요.']
    });
  }

  res.status(500).json({
    success: false,
    error: 'INTERNAL_SERVER_ERROR',
    message: '서버 내부 오류가 발생했습니다.',
    details: process.env.NODE_ENV === 'development' ? error.message : undefined
  });
});

// 서버 시작
app.listen(PORT, () => {
  console.log(`=================================`);
  console.log(`🚀 Todo Backend API Server`);
  console.log(`=================================`);
  console.log(`📍 서버 주소: http://localhost:${PORT}`);
  console.log(`📡 API 베이스: http://localhost:${PORT}/api`);
  console.log(`📁 정적 파일: http://localhost:${PORT}/public`);
  console.log(`🕐 시작 시간: ${new Date().toLocaleString('ko-KR')}`);
  console.log(`=================================`);
});

// 서버 종료 시 정리 작업
process.on('SIGINT', () => {
  console.log('\n서버를 종료합니다...');
  process.exit(0);
});

module.exports = app;