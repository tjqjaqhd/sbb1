# PostgreSQL 자동 설치 스크립트
# 관리자 권한으로 PowerShell에서 실행하세요

Write-Host "========================================" -ForegroundColor Green
Write-Host "PostgreSQL 자동 설치 스크립트" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

# 관리자 권한 확인
if (-NOT ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Host "이 스크립트는 관리자 권한이 필요합니다." -ForegroundColor Red
    Write-Host "PowerShell을 '관리자로 실행'으로 열어주세요." -ForegroundColor Red
    pause
    exit
}

Write-Host "1단계: PostgreSQL 다운로드 중..." -ForegroundColor Yellow

# PostgreSQL 14 다운로드 URL
$postgresUrl = "https://get.enterprisedb.com/postgresql/postgresql-14.15-1-windows-x64.exe"
$downloadPath = "$env:TEMP\postgresql-installer.exe"

try {
    # 다운로드
    Invoke-WebRequest -Uri $postgresUrl -OutFile $downloadPath -UseBasicParsing
    Write-Host "다운로드 완료!" -ForegroundColor Green

    Write-Host "2단계: PostgreSQL 설치 중..." -ForegroundColor Yellow

    # 무인 설치 실행
    $arguments = @(
        "--mode", "unattended",
        "--superpassword", "*tj1748426",
        "--servicename", "postgresql",
        "--serviceaccount", "postgres",
        "--servicepassword", "*tj1748426",
        "--serverport", "5432",
        "--locale", "Korean, Korea"
    )

    Start-Process -FilePath $downloadPath -ArgumentList $arguments -Wait -NoNewWindow

    Write-Host "3단계: 환경 변수 설정 중..." -ForegroundColor Yellow

    # PostgreSQL bin 디렉토리를 PATH에 추가
    $postgresPath = "C:\Program Files\PostgreSQL\14\bin"
    $currentPath = [System.Environment]::GetEnvironmentVariable("Path", "Machine")

    if ($currentPath -notlike "*$postgresPath*") {
        $newPath = $currentPath + ";" + $postgresPath
        [System.Environment]::SetEnvironmentVariable("Path", $newPath, "Machine")
        Write-Host "환경 변수 PATH에 PostgreSQL 추가됨" -ForegroundColor Green
    }

    Write-Host "4단계: 데이터베이스 및 사용자 생성 중..." -ForegroundColor Yellow

    # 잠깐 대기 (서비스 시작을 위해)
    Start-Sleep -Seconds 10

    # psql 명령으로 데이터베이스 생성
    $env:PGPASSWORD = "*tj1748426"

    & "C:\Program Files\PostgreSQL\14\bin\psql.exe" -U postgres -c "CREATE DATABASE bithumb_trading;"
    & "C:\Program Files\PostgreSQL\14\bin\psql.exe" -U postgres -c "CREATE USER tjqjaqhd WITH ENCRYPTED PASSWORD '*tj1748426';"
    & "C:\Program Files\PostgreSQL\14\bin\psql.exe" -U postgres -c "GRANT ALL PRIVILEGES ON DATABASE bithumb_trading TO tjqjaqhd;"
    & "C:\Program Files\PostgreSQL\14\bin\psql.exe" -U postgres -c "ALTER USER tjqjaqhd CREATEDB;"

    # 임시 파일 삭제
    Remove-Item -Path $downloadPath -Force

    Write-Host "========================================" -ForegroundColor Green
    Write-Host "PostgreSQL 설치 및 설정 완료!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "데이터베이스: bithumb_trading" -ForegroundColor Cyan
    Write-Host "사용자: tjqjaqhd" -ForegroundColor Cyan
    Write-Host "비밀번호: *tj1748426" -ForegroundColor Cyan
    Write-Host "포트: 5432" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Green

} catch {
    Write-Host "설치 중 오류 발생: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "수동 설치를 진행해주세요." -ForegroundColor Yellow
}

Write-Host "아무 키나 눌러 종료하세요..."
pause