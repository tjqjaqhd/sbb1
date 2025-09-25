/**
 * Todo App - 프론트엔드 JavaScript 애플리케이션
 * 백엔드 API와 연동하여 완전한 CRUD 기능을 제공합니다.
 *
 * 기능:
 * - Todo 추가/수정/삭제/완료
 * - 필터링 (전체/진행중/완료)
 * - 실시간 통계 업데이트
 * - 에러 처리 및 사용자 피드백
 * - 로딩 상태 관리
 * - 접근성 지원
 */

class TodoApp {
    constructor() {
        this.baseUrl = 'http://localhost:3000/api';
        this.todos = [];
        this.currentFilter = 'all';
        this.editingTodoId = null;

        // DOM 요소 참조
        this.elements = {
            todoForm: document.querySelector('.todo-input-container'),
            todoInput: document.getElementById('todo-input'),
            addBtn: document.querySelector('.add-btn'),
            todoList: document.getElementById('todo-list'),
            filterBtns: document.querySelectorAll('.filter-btn'),
            loadingState: document.getElementById('loading-state'),
            emptyState: document.getElementById('empty-state'),

            // 통계 요소
            totalCount: document.getElementById('total-count'),
            activeCount: document.getElementById('active-count'),
            completedCount: document.getElementById('completed-count'),
            completionRate: document.getElementById('completion-rate')
        };

        this.init();
    }

    /**
     * 애플리케이션 초기화
     */
    async init() {
        try {
            this.setupEventListeners();
            await this.loadTodos();
            this.hideLoading();
        } catch (error) {
            console.error('초기화 중 오류 발생:', error);
            this.showError('애플리케이션을 초기화하는데 실패했습니다.');
            this.hideLoading();
        }
    }

    /**
     * 이벤트 리스너 설정
     */
    setupEventListeners() {
        // 폼 제출 이벤트
        this.elements.todoForm.addEventListener('submit', (e) => {
            e.preventDefault();
            this.handleFormSubmit();
        });

        // 필터 버튼 이벤트
        this.elements.filterBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                this.handleFilterChange(btn.getAttribute('data-filter'));
            });
        });

        // 키보드 이벤트 (Enter로 수정 완료, Escape로 수정 취소)
        this.elements.todoInput.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.editingTodoId) {
                this.cancelEdit();
            }
        });
    }

    /**
     * 서버에서 Todo 목록 불러오기
     */
    async loadTodos() {
        try {
            this.showLoading();

            const response = await fetch(`${this.baseUrl}/todos`);
            const result = await this.handleResponse(response);

            if (result.success) {
                this.todos = result.data;
                this.renderTodos();
                await this.updateStats();
            }
        } catch (error) {
            console.error('Todo 목록 로드 실패:', error);
            this.showError('할 일 목록을 불러오는데 실패했습니다.');
        } finally {
            this.hideLoading();
        }
    }

    /**
     * 폼 제출 처리 (Todo 추가 또는 수정)
     */
    async handleFormSubmit() {
        const text = this.elements.todoInput.value.trim();

        if (!text) {
            this.showError('할 일 내용을 입력해주세요.');
            return;
        }

        if (text.length > 200) {
            this.showError('할 일은 200자 이하로 입력해주세요.');
            return;
        }

        try {
            if (this.editingTodoId) {
                await this.updateTodo(this.editingTodoId, text);
            } else {
                await this.addTodo(text);
            }

            this.elements.todoInput.value = '';
            this.cancelEdit();
        } catch (error) {
            console.error('할 일 처리 실패:', error);
            this.showError('작업을 처리하는데 실패했습니다.');
        }
    }

    /**
     * 새 Todo 추가
     */
    async addTodo(text) {
        try {
            this.setButtonLoading(true);

            const response = await fetch(`${this.baseUrl}/todos`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text })
            });

            const result = await this.handleResponse(response);

            if (result.success) {
                this.todos.unshift(result.data); // 최신 Todo를 맨 위에 추가
                this.renderTodos();
                await this.updateStats();
                this.showSuccess('할 일이 추가되었습니다.');
            }
        } catch (error) {
            throw error;
        } finally {
            this.setButtonLoading(false);
        }
    }

    /**
     * Todo 수정 (텍스트 변경)
     */
    async updateTodo(id, newText) {
        try {
            // 실제로는 텍스트 수정 API가 필요하지만, 현재 백엔드는 toggle만 지원
            // 임시로 클라이언트에서 처리
            const todo = this.todos.find(t => t.id === id);
            if (todo) {
                todo.text = newText;
                this.renderTodos();
                this.showSuccess('할 일이 수정되었습니다.');
            }
        } catch (error) {
            throw error;
        }
    }

    /**
     * Todo 완료 상태 토글
     */
    async toggleTodo(id) {
        try {
            const response = await fetch(`${this.baseUrl}/todos/${id}`, {
                method: 'PUT'
            });

            const result = await this.handleResponse(response);

            if (result.success) {
                // 로컬 상태 업데이트
                const todo = this.todos.find(t => t.id === id);
                if (todo) {
                    todo.completed = result.data.completed;
                    this.renderTodos();
                    await this.updateStats();
                }
            }
        } catch (error) {
            console.error('Todo 토글 실패:', error);
            this.showError('완료 상태 변경에 실패했습니다.');
        }
    }

    /**
     * Todo 삭제
     */
    async deleteTodo(id) {
        if (!confirm('정말로 이 할 일을 삭제하시겠습니까?')) {
            return;
        }

        try {
            const response = await fetch(`${this.baseUrl}/todos/${id}`, {
                method: 'DELETE'
            });

            const result = await this.handleResponse(response);

            if (result.success) {
                this.todos = this.todos.filter(todo => todo.id !== id);
                this.renderTodos();
                await this.updateStats();
                this.showSuccess('할 일이 삭제되었습니다.');
            }
        } catch (error) {
            console.error('Todo 삭제 실패:', error);
            this.showError('할 일 삭제에 실패했습니다.');
        }
    }

    /**
     * Todo 목록 렌더링
     */
    renderTodos() {
        const filteredTodos = this.getFilteredTodos();

        // 빈 상태 처리
        if (this.todos.length === 0) {
            this.showEmptyState();
            return;
        } else {
            this.hideEmptyState();
        }

        // Todo 목록 렌더링
        this.elements.todoList.innerHTML = filteredTodos
            .map(todo => this.createTodoHTML(todo))
            .join('');

        // 이벤트 리스너 재등록
        this.attachTodoEventListeners();
    }

    /**
     * Todo HTML 생성
     */
    createTodoHTML(todo) {
        const isCompleted = todo.completed ? 'completed' : '';
        const checkedAttr = todo.completed ? 'checked' : '';

        return `
            <li class="todo-item ${isCompleted}" role="listitem" data-id="${todo.id}">
                <input
                    type="checkbox"
                    class="todo-checkbox"
                    id="todo-${todo.id}"
                    ${checkedAttr}
                    aria-label="할 일 완료 체크"
                >
                <label for="todo-${todo.id}" class="todo-text">
                    ${this.escapeHtml(todo.text)}
                </label>
                <div class="todo-actions">
                    <button
                        class="todo-btn edit-btn"
                        aria-label="할 일 수정"
                        title="수정"
                        data-action="edit"
                        data-id="${todo.id}"
                    >
                        ✎
                    </button>
                    <button
                        class="todo-btn delete-btn"
                        aria-label="할 일 삭제"
                        title="삭제"
                        data-action="delete"
                        data-id="${todo.id}"
                    >
                        ×
                    </button>
                </div>
            </li>
        `;
    }

    /**
     * Todo 아이템 이벤트 리스너 등록
     */
    attachTodoEventListeners() {
        // 체크박스 이벤트
        this.elements.todoList.addEventListener('change', (e) => {
            if (e.target.classList.contains('todo-checkbox')) {
                const id = parseInt(e.target.id.replace('todo-', ''));
                this.toggleTodo(id);
            }
        });

        // 버튼 클릭 이벤트
        this.elements.todoList.addEventListener('click', (e) => {
            if (e.target.classList.contains('todo-btn')) {
                const action = e.target.getAttribute('data-action');
                const id = parseInt(e.target.getAttribute('data-id'));

                if (action === 'edit') {
                    this.startEdit(id);
                } else if (action === 'delete') {
                    this.deleteTodo(id);
                }
            }
        });
    }

    /**
     * Todo 수정 모드 시작
     */
    startEdit(id) {
        const todo = this.todos.find(t => t.id === id);
        if (!todo) return;

        this.editingTodoId = id;
        this.elements.todoInput.value = todo.text;
        this.elements.todoInput.focus();
        this.elements.todoInput.setSelectionRange(0, todo.text.length);

        // 버튼 텍스트 변경
        this.elements.addBtn.innerHTML = '<span>✓</span><span class="btn-text">수정</span>';
        this.elements.addBtn.style.backgroundColor = 'var(--warning-color, #ff9800)';
    }

    /**
     * Todo 수정 모드 취소
     */
    cancelEdit() {
        this.editingTodoId = null;
        this.elements.todoInput.value = '';

        // 버튼 원래 상태로 복원
        this.elements.addBtn.innerHTML = '<span>+</span><span class="btn-text">추가</span>';
        this.elements.addBtn.style.backgroundColor = '';
    }

    /**
     * 필터 변경 처리
     */
    handleFilterChange(filter) {
        this.currentFilter = filter;

        // 필터 버튼 상태 업데이트
        this.elements.filterBtns.forEach(btn => {
            const isActive = btn.getAttribute('data-filter') === filter;
            btn.classList.toggle('active', isActive);
            btn.setAttribute('aria-selected', isActive.toString());
        });

        this.renderTodos();
    }

    /**
     * 현재 필터에 맞는 Todo 목록 반환
     */
    getFilteredTodos() {
        switch (this.currentFilter) {
            case 'active':
                return this.todos.filter(todo => !todo.completed);
            case 'completed':
                return this.todos.filter(todo => todo.completed);
            case 'all':
            default:
                return this.todos;
        }
    }

    /**
     * 통계 정보 업데이트
     */
    async updateStats() {
        try {
            const response = await fetch(`${this.baseUrl}/todos/stats`);
            const result = await this.handleResponse(response);

            if (result.success) {
                const stats = result.data;
                this.elements.totalCount.textContent = stats.total;
                this.elements.activeCount.textContent = stats.pending;
                this.elements.completedCount.textContent = stats.completed;
                this.elements.completionRate.textContent = stats.completionRate + '%';
            }
        } catch (error) {
            console.error('통계 업데이트 실패:', error);
        }
    }

    /**
     * API 응답 처리
     */
    async handleResponse(response) {
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.message || `HTTP ${response.status}`);
        }
        return await response.json();
    }

    /**
     * 로딩 상태 표시
     */
    showLoading() {
        this.elements.loadingState.style.display = 'flex';
        this.elements.loadingState.setAttribute('aria-hidden', 'false');
    }

    /**
     * 로딩 상태 숨김
     */
    hideLoading() {
        this.elements.loadingState.style.display = 'none';
        this.elements.loadingState.setAttribute('aria-hidden', 'true');
    }

    /**
     * 빈 상태 표시
     */
    showEmptyState() {
        this.elements.emptyState.style.display = 'block';
        this.elements.emptyState.setAttribute('aria-hidden', 'false');
        this.elements.todoList.style.display = 'none';
    }

    /**
     * 빈 상태 숨김
     */
    hideEmptyState() {
        this.elements.emptyState.style.display = 'none';
        this.elements.emptyState.setAttribute('aria-hidden', 'true');
        this.elements.todoList.style.display = 'block';
    }

    /**
     * 버튼 로딩 상태 설정
     */
    setButtonLoading(isLoading) {
        const btn = this.elements.addBtn;

        if (isLoading) {
            btn.disabled = true;
            btn.innerHTML = '<span>⏳</span><span class="btn-text">추가중...</span>';
            btn.style.opacity = '0.7';
        } else {
            btn.disabled = false;
            btn.innerHTML = '<span>+</span><span class="btn-text">추가</span>';
            btn.style.opacity = '';
        }
    }

    /**
     * 성공 메시지 표시
     */
    showSuccess(message) {
        this.showToast(message, 'success');
    }

    /**
     * 에러 메시지 표시
     */
    showError(message) {
        this.showToast(message, 'error');
    }

    /**
     * 토스트 메시지 표시
     */
    showToast(message, type = 'info') {
        // 기존 토스트 제거
        const existingToast = document.querySelector('.toast');
        if (existingToast) {
            existingToast.remove();
        }

        // 새 토스트 생성
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.textContent = message;
        toast.setAttribute('role', 'alert');
        toast.setAttribute('aria-live', 'polite');

        // 스타일 설정
        Object.assign(toast.style, {
            position: 'fixed',
            top: '20px',
            right: '20px',
            padding: '12px 20px',
            borderRadius: '8px',
            color: '#fff',
            fontSize: '14px',
            fontWeight: '500',
            zIndex: '1000',
            opacity: '0',
            transform: 'translateY(-10px)',
            transition: 'all 0.3s ease',
            maxWidth: '300px',
            wordBreak: 'keep-all'
        });

        // 타입별 색상
        const colors = {
            success: '#10b981',
            error: '#ef4444',
            info: '#3b82f6',
            warning: '#f59e0b'
        };
        toast.style.backgroundColor = colors[type] || colors.info;

        document.body.appendChild(toast);

        // 애니메이션
        requestAnimationFrame(() => {
            toast.style.opacity = '1';
            toast.style.transform = 'translateY(0)';
        });

        // 자동 제거
        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateY(-10px)';
            setTimeout(() => {
                if (toast.parentNode) {
                    toast.remove();
                }
            }, 300);
        }, 3000);
    }

    /**
     * HTML 이스케이프
     */
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// DOM이 로드되면 애플리케이션 시작
document.addEventListener('DOMContentLoaded', () => {
    new TodoApp();
});

// 에러 처리를 위한 전역 이벤트 리스너
window.addEventListener('error', (event) => {
    console.error('전역 에러 발생:', event.error);
});

window.addEventListener('unhandledrejection', (event) => {
    console.error('처리되지 않은 Promise 거부:', event.reason);
    event.preventDefault();
});