# Research Lab Blog

ML/AI 연구 노트, 논문 리뷰, 코드 구현 기록을 위한 블로그.

Hugo + GitHub Pages로 운영. 마크다운을 push하면 자동 배포.

---

## 셋업 가이드 (최초 1회)

### 1. GitHub 레포 생성

GitHub에서 새 레포를 만든다:
- Repository name: `blog` (또는 원하는 이름)
- Public으로 설정
- README는 추가하지 않음

### 2. 로컬에 Hugo 설치

```bash
# macOS
brew install hugo

# Linux (Ubuntu/Debian)
sudo apt install hugo

# 또는 최신 버전 직접 다운로드
# https://github.com/gohugoio/hugo/releases
```

### 3. 이 프로젝트를 로컬에 클론

```bash
# 이 폴더의 파일들을 레포에 올리기
cd blog
git init
git remote add origin https://github.com/YOUR_USERNAME/blog.git
```

### 4. PaperMod 테마 설치

```bash
git submodule add --depth=1 https://github.com/adityatelange/hugo-PaperMod.git themes/PaperMod
```

### 5. 설정 수정

`hugo.toml`에서 다음을 변경:
```toml
baseURL = "https://YOUR_USERNAME.github.io/blog/"
```

`params.socialIcons`의 GitHub URL도 변경.

### 6. 로컬 미리보기

```bash
hugo server -D
# http://localhost:1313/blog/ 에서 확인
```

### 7. GitHub에 push

```bash
git add .
git commit -m "Initial blog setup"
git branch -M main
git push -u origin main
```

### 8. GitHub Pages 활성화

1. GitHub 레포 → Settings → Pages
2. Source를 **GitHub Actions**로 선택
3. 자동으로 `.github/workflows/deploy.yml`이 실행됨
4. 몇 분 후 `https://YOUR_USERNAME.github.io/blog/` 에서 확인

---

## 새 글 작성법

### 방법 1: 직접 마크다운 작성

```bash
# 새 글 파일 생성
hugo new posts/my-new-post.md
```

또는 `content/posts/` 폴더에 직접 `.md` 파일을 생성해도 된다.

### Front matter 템플릿

```yaml
---
title: "글 제목"
date: 2026-03-18
tags: ["연구노트", "SancMamba"]
categories: ["ML/AI"]
summary: "한줄 요약"
math: true        # 수식 사용 시
toc: true         # 목차 표시
draft: false      # true면 빌드에서 제외
---
```

### 방법 2: Claude에게 마크다운 받기

Claude와 대화하며 연구 내용을 정리한 뒤, 마크다운 파일을 받아 `content/posts/`에 넣고 push.

### 발행

```bash
git add content/posts/my-new-post.md
git commit -m "Add: 새 글 제목"
git push
```

push하면 GitHub Actions가 자동으로 빌드 + 배포한다.

---

## 수식 사용법

KaTeX가 설정되어 있으므로:

```markdown
인라인 수식: $E = mc^2$

블록 수식:
$$\mathcal{L} = \sum_{t} -\log P(x_{t+1} | x_{\leq t})$$
```

## 코드 블록

````markdown
```python
import torch
model = SancMamba(config)
```
````

## 이미지

```markdown
![설명](/images/my-image.png)
```

이미지 파일은 `static/images/`에 넣으면 된다.

---

## 폴더 구조

```
blog/
├── .github/workflows/deploy.yml   # 자동 배포
├── content/
│   ├── posts/                     # 블로그 글 (마크다운)
│   └── search.md                  # 검색 페이지
├── layouts/partials/
│   └── extend_head.html           # KaTeX 수식 지원
├── static/images/                 # 이미지 파일
├── themes/PaperMod/               # 테마 (git submodule)
└── hugo.toml                      # Hugo 설정
```
