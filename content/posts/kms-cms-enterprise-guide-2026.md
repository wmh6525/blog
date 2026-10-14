---
title: "[리서치] 기업은 지식과 콘텐츠를 어떻게 관리하나 — KMS·CMS의 제품·가격·실제 운영·지식 내재화·데이터 축적 (2026)"
date: 2026-10-14T09:00:00+09:00
tags: ["리서치", "KMS", "CMS", "지식관리", "RAG", "사내검색", "KCS", "SECI", "Copilot", "헤드리스CMS", "2026"]
categories: ["Engineering"]
summary: "회사 안의 지식을 다루는 KMS와 밖으로 내보낼 콘텐츠를 다루는 CMS를 쉽게 풀었다. Confluence·Copilot·Glean·국내 구축형 업체와 WordPress·Contentful·AEM의 가격과 판매 방식, Morgan Stanley·LinkedIn·코레일·한수원 등 실제 운영 사례, KCS로 지식을 쌓고 권한·청킹·HWP 문제를 풀어 AI 검색에 올리는 과정, SECI 모델과 AAR·명장 제도로 본 지식의 내재화, AI가 실력을 키우는지 줄이는지에 대한 2025~2026 연구까지."
math: false
toc: true
draft: false
---

<!-- LEARNING-START -->
## 처음 읽는 분을 위한 안내

**학습 위치:** [앱을 만들고 운영하는 사례]({{< relref "learn" >}}#engineering) · 7/7

회사가 가진 지식과 콘텐츠를 어떤 제품으로, 얼마를 내고, 어떻게 관리하는지 다룬 리서치다. 기술 지식이 없어도 읽을 수 있다. 문서가 만들어져 검색되고, 사람의 실력이 되기까지의 흐름을 따라가면 된다.

**먼저 알아둘 말**

- [RAG]({{< relref "ai-glossary" >}}#rag): 질문과 관련된 문서를 먼저 찾아 그 내용을 근거로 답을 만드는 방식이다.
- [DB·계정·권한]({{< relref "ai-glossary" >}}#db): 누가 어떤 데이터를 읽고 바꿀 수 있는지 정하는 장치다. 사내 AI 검색에서는 이 권한이 가장 중요하다.
- [그래프·지식 그래프]({{< relref "ai-glossary" >}}#graph): 문서·사람·사건을 점으로, 그 사이 관계를 선으로 이어 저장한 구조다.

**이번 글의 목표:** 지식이 기록·검색·체득되는 과정과 그 비용 구조 이해하기.

**이 학습 경로의 앞 글:** [\[대회\] 공공조달 법령위반 판정 — 추출·결정 분리, 법정 상수, 시간 예산 설계 (DACON 236754)]({{< relref "posts/dacon-procurement-law-violation" >}}).

검색으로 자료를 가져오는 원리는 [RAG 전체 흐름]({{< relref "posts/rag-survey-2026" >}})에서 먼저 볼 수 있다.

**본문에서 먼저 볼 순서:** **1. 먼저 결론부터** → **2. KMS와 CMS, 무엇이 다른가** → **7. 데이터는 어떻게 쌓이나** → **8. 지식의 내재화**.

첫 독서에서는 표의 가격보다 “무엇을 사고, 어떻게 쌓고, 누가 배우는가”의 흐름을 잡자. 제품별 가격과 출처는 실제로 비교할 때 다시 확인해도 된다.

<!-- LEARNING-START-END -->

---

> **작성 기준일: 2026년 10월 14일.** 공개된 공식 문서·가격표·보도·논문에서 확인한 내용만 적었다. 출처 번호는 글 끝 [참고 자료](#참고-자료)에 모았다. 공식 페이지를 직접 열지 못하고 검색 요약이나 제3자 자료로만 확인한 수치는 문장 안에 따로 표시했다. 여러 출처를 묶어 내린 판단은 “정리”라고 밝혔다. 가격은 확인한 날 기준이며 지역·계약 조건에 따라 달라진다.

## 1. 먼저 결론부터

이 글은 앞서 정리한 「기업의 KMS·CMS 관리 방식 리서치」를 넓힌 글이다. 질문은 다섯 개다. KMS와 CMS는 각각 무엇인가? 누가 무엇을 어떤 방식으로 파는가? 기업은 실제로 어떻게 쓰는가? 지식은 어떻게 사람의 실력으로 바뀌는가? 그 밑에 깔리는 데이터는 어떻게 쌓이는가?

| 질문 | 짧은 답 |
|---|---|
| 무엇이 다른가 | KMS는 **직원이 일하면서 찾아 쓰는 지식**을, CMS는 **고객에게 보여 줄 웹·앱 콘텐츠**를 관리한다. 사내 위키는 둘 사이에 걸쳐 있다. |
| 어떻게 파나 | 해외 SaaS는 **사용자 1명당 월 구독료**가 기본이다. 여기에 AI 기능을 상위 요금제에 묶거나, 좌석당 추가 요금을 받거나, 쓴 만큼 크레딧으로 청구한다. 대기업용 상위 요금제는 대부분 견적제다. 국내 공공·금융 시장은 **구축형(온프레미스) 라이선스 + 유지보수**나 SI 구축 사업이 중심이다. |
| 실제로 어떻게 쓰나 | 최근 흐름은 기존 문서 저장소 위에 **AI 검색**(RAG)을 얹는 방식이다. 가장 큰 걸림돌은 모델 성능보다 **권한 정리**다. |
| 지식은 어떻게 내재화되나 | 문서를 읽기만 해서는 실력이 되지 않는다. 직접 해 보고, 돌아보고, 함께 일하는 과정이 필요하다. AI가 신입의 성과를 끌어올린다는 연구와, AI에 기대면 실력이 덜 쌓인다는 연구가 함께 나와 있다. |
| 데이터는 어떻게 쌓이나 | 일하는 도중에 기록하고, 메타데이터와 담당자를 붙이고, 원본 권한과 함께 색인하고, 쓰이지 않는 문서는 보관 처리하고, 사용 데이터로 품질을 잰다. |

## 2. KMS와 CMS, 무엇이 다른가

비유로 시작해 보자. **KMS**(지식관리시스템)는 회사 안의 도서관이자 선배의 업무 노트다. 규정·매뉴얼·장애 처리 기록·제안서·회의록처럼 “일하다가 막혔을 때 찾아보는 것”이 들어 있다. 읽는 사람은 직원이다.

**CMS**(콘텐츠관리시스템)는 회사의 쇼윈도와 출판 설비에 가깝다. 홈페이지 상품 소개, 공지, 블로그, 앱 화면 문구처럼 “밖에 내보낼 것”을 만들고, 승인받고, 게시한다. 읽는 사람은 주로 고객이다.

| 구분 | KMS | CMS |
|---|---|---|
| 담는 것 | 사내 지식·노하우·규정·매뉴얼·사례 | 웹사이트·앱·채널에 게시하는 콘텐츠 |
| 주로 쓰는 사람 | 임직원, 상담·지원 조직, 현업 전문가 | 마케팅·콘텐츠·웹 운영 조직, 개발자 |
| 관리의 핵심 | 지식이 만들어지고 다시 쓰이는지, 최신인지, 누가 볼 수 있는지 | 콘텐츠 구조, 승인 절차, 게시, 채널 배포 |
| 대표 제품 | Confluence, SharePoint·Microsoft 365 Copilot, Notion, Glean, 자체 구축 RAG | WordPress, Shopify, Wix, Contentful, Adobe Experience Manager(AEM), Sitecore |

경계가 흐린 곳도 있다. W3Techs는 CMS를 “웹사이트 콘텐츠를 만들고 관리하는 애플리케이션”으로 정의하고 위키·블로그 엔진·게시판까지 포함해 센다. [22] 사내 위키는 KMS로 쓰이지만 기술로는 CMS와 같은 뿌리다. 고객센터의 도움말 페이지도 상담원에게는 KMS이고 고객에게는 공개 콘텐츠다.

## 3. KMS 시장: 누가 무엇을 파나

KMS 제품은 크게 네 갈래로 나눌 수 있다. 문서를 쓰고 쌓는 **위키·협업형**, 여러 시스템을 한 번에 찾아 주는 **AI 사내 검색형**, 고객 응대와 묶인 **서비스 지식형**, 그리고 국내 공공·금융 시장의 **구축형**이다.

### 3.1 위키·협업형: 문서를 쓰고 쌓는 곳

**Atlassian Confluence.** 클라우드 요금제는 Free(10명까지 무료), Standard(사용자당 월 $5.78), Premium(사용자당 월 $11.16), Enterprise(견적, 801명~25만 명)로 나뉜다. [K1] AI 기능인 Rovo는 Standard 이상에 기본으로 들어 있어 따로 사지 않아도 된다. 대신 요금제별로 사용자당 매달 쓸 수 있는 크레딧이 정해져 있다(Standard 25, Premium 70, Enterprise 150). 크레딧은 조직 전체가 함께 쓰고 넘치면 1크레딧에 $0.01을 낸다. 초과 과금은 기본으로 켜져 있고 관리자가 정한 상한에서 멈춘다. [K2]

**Microsoft SharePoint + Microsoft 365 Copilot.** 많은 기업에서 SharePoint를 사실상 문서 저장소 겸 KMS로 쓴다. Copilot은 기본 Microsoft 365 요금제 위에 얹는 추가 라이선스로 팔린다. 웹 검색 기반 Copilot Chat은 대상 구독에 무료로 포함되지만 회사 데이터를 근거로 답하려면 Copilot 라이선스가 필요하다. [K3] 2026년 5월에는 E5·Copilot·Agent 365·Entra Suite를 묶은 **Microsoft 365 E7**이 사용자당 월 $99(연간 약정)로 나왔다. [K4] 단독 Copilot 가격은 기업용 사용자당 월 $30, 300명 이하용 Copilot Business 월 $21로 알려져 있지만 이번에는 제3자 자료로만 확인했다. [K5] 권한·사이트 관리 기능을 모은 SharePoint Advanced Management는 단독 구매 시 사용자당 월 $3이다(검색 요약으로 확인). [K6]

**Notion.** 한국 원화 가격으로 Free, Plus(멤버당 월 14,000원), Business(월 30,000원), Enterprise(견적)이다. AI 에이전트·회의록·사내 검색은 Business부터 기본 포함이다. 직접 만든 Custom Agent는 따로 크레딧(월 1,000크레딧당 $10)으로 청구한다. [K7]

### 3.2 AI 사내 검색형: 흩어진 시스템을 한 번에 찾는 곳

회사 지식은 한 곳에 있지 않다. 위키·메일·메신저·티켓·드라이브에 흩어져 있다. 이 갈래는 문서를 새로 쓰게 하기보다 이미 있는 시스템에 커넥터를 꽂아 한꺼번에 색인하고 AI로 답하게 한다.

| 제품 | 무엇을 하나 | 가격 공개 여부 |
|---|---|---|
| Glean | 275개 이상 커넥터로 사내 데이터를 색인하고 지식 그래프를 만든다. 원본 시스템의 권한을 그대로 물려받아 적용한다고 밝힌다. [K8] | 공개하지 않음. 경쟁사 블로그의 추정치만 있다. |
| Guru | AI 지식 계층·지식 에이전트·100개 이상 연동. 스스로를 “좌석당 도구가 아니라 플랫폼과 전문 지원을 함께 파는 솔루션”이라고 설명한다. [K9] | 고객별 견적 |
| Google Gemini Enterprise | 옛 Agentspace를 흡수했다. Standard·Plus가 좌석당 월 $30부터, Gemini Business가 월 $21(연간 약정). [K10] | 공개(보도 기준) |
| Amazon Q Business | Lite 사용자당 월 $3, Pro 월 $20에 색인 사용 시간 요금이 따로 붙는다. AWS는 Quick Suite를 후속 제품으로 안내하고 신규 고객은 더 받지 않는다. [K11][D9] | 공개 |
| Moveworks → ServiceNow | ServiceNow가 28억 5천만 달러에 인수해 2025년 12월 15일 완료했다. Moveworks 직원 사용자는 550만 명이다. [K12][K13] | 공개하지 않음 |

### 3.3 서비스 지식형: 고객 응대와 묶인 지식베이스

콜센터·IT 헬프데스크는 KMS를 가장 치열하게 쓰는 곳이다. 이 분야 제품은 지식베이스를 **상담원 좌석에 묶어서** 판다.

- **Zendesk:** 도움말 센터가 상담원 좌석에 포함된다. Suite Team은 상담원당 월 $55(연간)에 도움말 센터 1개, Professional은 $115에 최대 5개다. AI 상담원은 기본 포함이지만 **자동으로 해결한 건수**에 따라 건당 $1.50(약정)~$2.00를 받는다. Copilot 추가 기능은 상담원당 월 $50이다. [K14]
- **Salesforce Service Cloud:** 사용자당 월 $25(Starter)부터 $550(Max)까지다. Max에는 Agentforce for Service가 모두 포함된다. [K15]
- **ServiceNow:** 공개 가격표가 없다.

지식 작성 방법론인 KCS(뒤의 7장 참고)를 인증하는 Consortium for Service Innovation의 2023년 목록에는 Salesforce와 ServiceNow가 올라 있다. [K16]

### 3.4 국내 시장: 구축형과 공공 사업이 중심

국내 업체는 사내 검색엔진에서 출발해 RAG·에이전트로 넓혀 왔다. 고객 다수가 공공·국방·금융이라 **외부 인터넷과 분리된 회사 서버에 설치하는 구축형**이 많다.

| 업체 | 파는 것 | 최근 확인된 사업·특징 |
|---|---|---|
| 코난테크놀로지 | Konan Search, Konan RAG-X, Agent-X 등 | 2026년 6월 법제처 '생성형 AI 법령검색 시스템 구축' 사업을 65.5억 원(550일)에 따냈다. 4개사 컨소시엄의 주관사이며 약 760만 건의 법령 데이터를 벡터 DB로 만든다. [K17] |
| 솔트룩스 | LUXIA LLM, LUXIA A.RAG | 2025년 중앙노동위원회 'AI 디지털노동위원회' 1차 사업(15억 원, 3년 36억 원)을 수주했다. [K18] 2026년 3월에는 GPU 서버에 소프트웨어를 미리 설치한 장비형 제품 'LUXIA-ON'을 내놓았다. [K19] |
| 와이즈넛 | 검색, 챗봇, WISE iRAG | 라이선스 판매가 중심이다. 보도된 평균 사업 규모는 검색 4~5천만 원, 챗봇 약 5억 원, 생성형 AI 10~15억 원이다. [K20] |
| 사이냅소프트 | 도큐애널라이저 | HWP·HWPX·오피스·PDF 문서를 RAG에 넣기 좋은 Markdown·JSON으로 바꾼다(검색 요약으로 확인). [K21] 법제처 컨소시엄에도 참여했다. [K17] |
| 삼성SDS | Brity Copilot, FabriX | 삼성SDS·네이버클라우드 컨소시엄이 '범정부 AI 공통기반' 사업을 맡았다(1차 약 90억 원). [K22] 2026년 10월 기준 11개 LLM을 올려 시범 운영 중이다. [K23] |
| 네이버클라우드 | 뉴로클라우드 for HyperCLOVA X | 클라우드 LLM을 쓰기 어려운 공공·금융 고객의 데이터센터에 직접 설치하는 구축형 LLM(검색 요약으로 확인). [K24] |
| LG CNS | AI·클라우드 사업 | 2026년 1분기 AI·클라우드 매출이 7,654억 원으로 전체의 약 58%다. ChatGPT Enterprise도 재판매한다. [K25] |

**왜 국내는 구축형이 많을까.** 공공기관은 오랫동안 업무망을 인터넷과 물리적으로 나누는 망분리 원칙 아래 있었고 민간 클라우드를 쓰려면 클라우드 보안인증(CSAP)이 필요했다. 이 틀이 바뀌는 중이다. 국가정보원은 2025년 1월 일률적 망분리 대신 데이터를 C·S·O 세 등급으로 나누는 국가 망보안체계(N²SF) 초안을 냈다. [K26] 2026년 4월에는 공공 클라우드 검증을 국정원 절차로 일원화하고 CSAP는 민간 자율 인증으로 바꾼다고 발표했다. 1년 유예 뒤 2027년 하반기에 전면 시행된다. [K27] 2026년 9월 나온 방침에 따르면 공개(O) 등급 데이터만 다루는 상용 생성형 AI는 검증을 면제받을 수 있다. [K28] 민감하거나 기밀인 업무는 여전히 검증 대상이라 당분간은 구축형과 클라우드가 함께 갈 가능성이 크다(정리).

## 4. KMS는 어떤 방식으로 팔리나

앞의 제품들을 판매 방식으로 다시 묶으면 다섯 가지가 나온다.

| 판매 방식 | 어떻게 돈을 받나 | 예 |
|---|---|---|
| 좌석 구독 + AI 상위 요금제 포함 | 사용자 1명당 월 요금. AI는 비싼 요금제에 기본 포함 | Notion Business, Confluence + Rovo, Zendesk Suite |
| 좌석당 AI 추가 요금 | 기본 요금제 위에 AI 라이선스를 1명씩 더 산다 | Microsoft 365 Copilot(E7은 번들), Zendesk Copilot |
| 사용량·크레딧 | 쓴 만큼 낸다. 좌석 요금과 섞이는 경우가 많다 | Rovo 크레딧, Notion 에이전트 크레딧, Amazon Q 색인 시간, Zendesk 자동 해결 건당 요금 |
| 견적형 엔터프라이즈 | 가격을 공개하지 않고 영업 협상 | Glean, Guru, ServiceNow, 각 제품의 Enterprise 요금제 |
| 국내 구축형 | 라이선스 + 연간 유지보수, SI 구축 사업, 장비형 제품 | 법제처 65.5억 원, 중노위 36억 원, LUXIA-ON |

유지보수 요율은 외국 소프트웨어가 연간 라이선스 금액의 20~22% 수준이고 국산은 저가 입찰 때문에 그만큼 받지 못한다는 보도가 있다. [K29]

**정리.** 2025~2026년 해외 제품의 공통점은 AI를 따로 팔던 방식에서 상위 요금제에 묶는 방식으로 옮겨 가면서 많이 쓰는 고객에게는 크레딧으로 추가 요금을 받는다는 점이다. 그래서 도입할 때는 좌석 수와 함께 “AI를 얼마나 쓸 것인가”까지 예상해야 비용이 맞는다.

## 5. CMS 시장: 종류, 가격, 판매 방식

### 5.1 CMS의 다섯 가지 종류

| 종류 | 뜻 | 예 |
|---|---|---|
| 전통형(결합형) CMS | 콘텐츠 편집과 화면 출력이 한 프로그램 안에 묶여 있다 | WordPress, Drupal, 그누보드, 라이믹스 |
| SaaS 사이트 빌더 | 설치 없이 월 요금을 내고 쓰는 홈페이지·쇼핑몰 제작 서비스 | Wix, Squarespace, Shopify |
| 헤드리스 CMS | 화면은 없고 콘텐츠만 API로 내보낸다. 웹·앱·키오스크가 같은 콘텐츠를 가져다 쓴다 | Contentful, Sanity, Strapi, Storyblok |
| DXP(디지털 경험 플랫폼) | CMS에 개인화·분석·마케팅 도구를 묶은 대기업용 묶음 | Adobe Experience Manager, Sitecore, Optimizely, Acquia |
| DAM(디지털 자산 관리) | 승인된 이미지·영상 같은 자산을 저장하고 관리한다. CMS는 DAM의 자산을 가져다 페이지를 만든다 [C1] | Bynder, Acquia DAM, AEM Assets |

“헤드리스”라는 말이 어렵다면 이렇게 생각하면 된다. 전통형 CMS는 식당의 주방과 홀이 붙어 있는 구조다. 헤드리스는 주방만 있고 음식(콘텐츠)을 배달 앱·매장·행사장 어디로든 내보낸다.

### 5.2 점유율: 공개 웹 기준

W3Techs의 2026년 10월 14일 자료를 보면 WordPress는 전체 웹사이트의 40.0%, CMS를 쓰는 사이트 중에서는 58.5%를 차지한다. Shopify는 7.8%, Wix는 6.2%, Squarespace는 3.6%다(CMS 사용 사이트 기준). 전체 웹사이트의 31.6%는 W3Techs가 추적하는 CMS를 쓰지 않는다. [C2] WordPress의 CMS 점유율은 2025년 7월 61.0%에서 꾸준히 내려왔다. [21] 다만 이 통계는 공개된 웹사이트의 겉모습으로 판별한다. 헤드리스 CMS는 화면에 흔적이 잘 남지 않아 실제보다 적게 잡히고 사내 시스템은 아예 빠진다.

### 5.3 누가 얼마에 파나

| 제품 | 가격 구조 (2026.10.14 확인) |
|---|---|
| WordPress | 소프트웨어는 GPL 오픈소스로 무료다. [C3] 대기업용 WordPress VIP는 세 가지 패키지 모두 “가격 문의”이며 연간 계약이다. 트래픽이 갑자기 몰려도 초과 요금은 받지 않는다. [C4] |
| Shopify | 월 $25 / $65 / $399(연간 결제 시 $19 / $49 / $299). 대기업용 Plus는 월 $2,300부터. [C5] |
| Wix | 무료부터 Business Elite 월 $159.77까지(연간 결제, 지역별 상이). Enterprise는 견적. [C6] |
| Squarespace | 월 17,500원 / 25,000원 / 38,000원(연간 결제). [C7] |
| Contentful | 무료(상업적 사용 불가), Lite 월 $300(사용자 20명, API 호출 100만 회), Premium 견적. 공식 페이지가 차단돼 검색 요약으로만 확인했다. [C8] |
| Sanity | 무료(20석), Growth 좌석당 월 $15, Enterprise 견적. API 요청이 넘치면 2만 5천 회당 $1. [C9] |
| Strapi | 핵심 CMS는 MIT 오픈소스라 직접 설치하면 무료다. 클라우드는 프로젝트당 월 $35 / $90 / $450. 승인 워크플로·감사 로그는 Enterprise 전용이다. [C10][C11] |
| Storyblok | 무료, Growth 월 $99, Growth Plus 월 $349, Premium 이상 견적. [C12] |
| AEM·Sitecore·Optimizely | 공개 가격 없음. Optimizely는 “모든 요금제를 개별 구성한다”고 밝힌다. [C13] AEM의 AI 에이전트는 별도 'Agentic SKU'로 판다. [C14] |

정리하면 CMS 시장은 **아래쪽은 공개 가격, 위쪽은 견적**이다. 작은 사업자는 월 몇만 원짜리 사이트 빌더를 쓰고 헤드리스 CMS는 무료로 시작해 사용자·API 호출·언어 수에 따라 요금이 오른다. DXP는 처음부터 영업 협상으로 판다.

### 5.4 CMS에 들어온 AI

2025~2026년에 실제 출시된 기능만 추리면 다음과 같다.

- **Contentful AI Actions:** 2025년 3월 31일 정식 출시. 번역·SEO·대체 텍스트 작성 같은 작업을 콘텐츠 운영 안에서 생성형 AI로 돌린다. 원하는 LLM을 연결할 수 있다. [C15]
- **Adobe AEM:** 2025년 9월 AI 에이전트를 정식 출시했다. 깨진 링크나 성과가 낮은 페이지를 찾는 Site Optimization Agent 등이다. [C16] 에이전트는 AEM 클라우드 서비스에서만 돌고 구축형 AEM 6.5에서는 쓸 수 없다. [C14]
- **Sitecore:** 2025년 11월 10일 SitecoreAI를 출시하고 XM Cloud 고객을 자동으로 옮겼다. 규정 준수를 이유로 AI를 끌 수 있다. [C17]
- **WordPress:** 2026년 5월 20일 나온 7.0에 AI 클라이언트와 외부 AI 연결 화면이 코어로 들어갔다. 이미지·제목·요약 생성은 선택형 플러그인이다. [C18]

## 6. 실제로는 어떻게 쓰고 있나

### 6.1 KMS 운영 사례

**Morgan Stanley(자산관리 부문).** 약 10만 건의 내부 문서를 GPT-4 위에 올리는 구상으로 시작했다. [11] 도입 전에 번역, 요약(정확성·관련성·일관성 기준), 전문가 답변과의 비교라는 세 가지 평가를 만들었다. 결과는 자문역의 98%가 매일 사용하고 문서 접근률이 20%에서 80%로 오른 것이다. [D1] OpenAI가 낸 고객 사례 자료다.

**LinkedIn 고객 서비스.** 과거 장애 티켓을 지식 그래프로 바꾸고 RAG와 합쳤다. 약 6개월 운영하며 팀을 무작위로 두 그룹으로 나눠 비교했더니 문제 하나를 해결하는 시간의 중앙값이 7시간에서 5시간으로 28.6% 줄었다. [D2] 데이터를 어떻게 쌓았는지는 7장에서 다룬다.

**국내 사례.**

| 기관 | 무엇을 했나 | 관리 방식 |
|---|---|---|
| 코레일 '에어파인더'(2026.2) | 부서별 지침·규정·매뉴얼·보고서를 검색·요약하는 AI 챗봇 [12] | 인터넷과 분리된 업무망에서만 동작한다. 부서별로 데이터를 나눠 저장하고, 인사정보와 연동한 인증으로 자기 부서 데이터만 보게 한다. |
| 한국수력원자력 '하이누리' | 건설·발전 운영·원자력 안전·수력/신재생 데이터를 학습한 RAG, 부서별 챗봇 [I1] | 하이퍼클로바X 기반으로 자체 데이터센터에 구축했다. |
| LG CNS | 사업 방법론·R&D 산출물 등 약 52만 페이지를 자연어로 검색 [13] | 전문가 지식을 모으는 지식마켓 '브레인즈'를 따로 운영한다. |
| 코오롱베니트 '코아이봇'(2024.8) | 규정·복리후생·인사제도 질의응답, 출처 파일 제공 [15] | 연동 범위를 사내 시스템으로 좁혀 환각을 줄였다. 약 1개월 베타테스트 뒤 정식 도입했다. |
| 범정부 AI 공통기반 | 정부 부처가 함께 쓰는 생성형 AI 기반 [K22][K23] | 행정망 안 민관협력 클라우드 구역에서 운영한다. |

**실패에서 배운 사례도 있다.** NASA는 1994년부터 교훈 데이터베이스 LLIS를 운영했다. [I2] 그런데 2012년 NASA 감사관실 보고서에 따르면 28개 프로젝트에서 이 시스템을 쓴 관리자는 57%, 교훈을 올린 관리자는 43%에 그쳤다. 관리자들은 “오래됐고, 쓰기 불편하고, 필요한 내용이 없다”고 답했고 대신 사내 잡지와 연례 세미나를 더 찾았다. 연간 약 75만 달러를 쓰는 데 대해 감사관실은 비용이 타당한지 물었다. [I3] 국내 정부통합지식행정시스템(GKMC)도 2014년 행정안전부 스스로 “등록 건수는 늘었지만 질은 높아지지 않았다”고 평가했다. [29] 두 사례 모두 시스템을 만들어 두면 사람들이 알아서 쓸 것이라는 기대가 어긋났다.

**권한이 가장 큰 걸림돌이다.** AI 검색은 사용자가 열어 볼 수 있는 모든 문서를 대상으로 한다. 그래서 그동안 숨어 있던 권한 설정 실수가 그대로 드러난다. Gartner 2024년 조사에서 IT 리더의 40%가 과잉 공유 우려로 Copilot 배포를 3개월 이상 미뤘다. [17] CoreView 2026년 조사에서는 66%가 배포를 중단하거나 취소했고 그중 73%가 기밀 노출 우려를 이유로 꼽았다. [19][20]

### 6.2 CMS 운영 사례

**대기업 CMS의 기본 구조.** AEM 클라우드 서비스를 예로 보면, 콘텐츠를 쓰는 **작성(Author) 계층**, 게시 전에 확인하는 **미리보기(Preview) 계층**, 방문자에게 내보내는 **게시(Publish) 계층**이 나뉘어 있다. 개발·스테이징·운영 환경도 따로 둔다. 스테이징은 운영과 같은 크기로 맞춘다. 게시된 콘텐츠는 Adobe가 관리하는 CDN으로 나간다. [C19]

**역할과 승인.** Contentful은 역할마다 읽기·쓰기 권한을 정하고 콘텐츠 구조(콘텐츠 타입)를 바꾸는 권한은 따로 떼어 둔다. 상위 요금제에서는 항목·필드·언어 단위까지 권한을 나눌 수 있고 운영 환경과 검증 환경 접근도 역할별로 정한다. [27][28] Strapi의 검토 워크플로는 단계마다 담당 역할을 지정하고 필수 단계를 거쳐야만 게시되게 할 수 있다. [C11]

**공개된 이전 사례(모두 업체가 낸 자료).**

- 삼성닷컴: 삼성SDS가 구축형 AEM과 오라클 DB를 AWS로 옮겼다. 전환 시간은 “1시간 미만”이었다. [C20]
- Marc O'Polo: Storyblok으로 옮기며 37개 시장·6개 언어를 운영한다. 프로토타입을 48시간 만에 만들었는데 예전 통합형 제품에서는 비슷한 작업에 4개월 가까이 걸렸다고 한다. [C21]
- Align: Contentful로 130개가 넘는 웹사이트를 운영하고 49개 시장에 8주 만에 출시했다(검색 요약으로 확인). [C22]

**국내 CMS 현장.** 커뮤니티·중소 사이트에서는 국산 오픈소스인 그누보드와 XE를 이은 라이믹스가 많이 쓰인다. 그누보드에는 KG이니시스·카카오페이 같은 국내 결제 연동이 들어 있다. [C23][C24] 공공기관 홈페이지에는 전자정부 표준프레임워크와 호환되는 국산 CMS가 팔린다. [C25] 공공 홈페이지는 웹 접근성을 지켜야 하는데 2026년 1월 22일부터 「디지털포용법」이 시행되며 관련 규정이 옮겨 갔다. 웹 접근성 품질마크는 과기정통부가 지정한 기관이 KWCAG 2.2 기준으로 심사한다. [C26][C27] 대기업 쪽에서는 LG CNS가 2023년 어도비코리아와 AEM 기반 디지털 채널 사업 협약을 맺었다. [C28]

## 7. 데이터는 어떻게 쌓이나

AI 사내 검색이 퍼지면서 “데이터를 어떻게 쌓느냐”가 KMS의 핵심 질문이 됐다. 지식이 쓰일 수 있는 상태가 되기까지는 대략 여섯 단계를 거친다.

```
① 수집      일하는 도중에 기록한다 (티켓·상담·장애 보고 → 지식 문서)
② 구조화    분류어·메타데이터·담당자·검토일을 붙인다
③ 색인      커넥터로 가져와 쪼개고(청킹) 벡터로 바꾼다
④ 권한      원본 시스템의 열람 권한을 문서·조각마다 함께 저장한다
⑤ 정리      오래되고 안 쓰이는 문서는 보관 처리하거나 지운다
⑥ 측정      재사용·검색 실패·답변 품질을 재고 다시 ①로 돌아간다
```

### 7.1 수집: 일이 끝난 뒤가 아니라 일하는 도중에

서비스 조직에서 가장 널리 쓰이는 방법론은 KCS(Knowledge-Centered Service)다. KCS는 지식 문서를 “따로 시간을 내서 쓰는 추가 업무가 아니라 일하다 생기는 부산물”로 본다. 요청을 다 처리한 뒤에 문서를 쓰면 중요한 정보를 이미 놓친 것이라고까지 말한다. 검색 자체도 기록의 한 형태로 보고 내용만큼 맥락이 중요하다고 강조한다. [D3]

KCS 문서는 정해진 틀을 따른다. [D4]

| 칸 | 내용 |
|---|---|
| 문제(Issue) | 요청자가 쓴 말 그대로 적는다 |
| 환경(Environment) | 제품·버전·최근 변경 사항 |
| 해결(Resolution) | 답이나 조치 단계 |
| 원인(Cause) | 선택 항목 |
| 메타데이터 | 날짜·재사용 횟수 같은 자동 값과 대상 독자·품질 같은 사람이 정하는 값 |

문서에는 신뢰도 상태도 붙는다. 해결책이 아직 없는 **작성 중**, 해결책은 있지만 확신이 낮은 **미검증**, 다시 써도 될 만큼 완성된 **검증됨**, 검색에서 빠지지만 되살릴 수 있는 **보관**이다. KCS는 “보관이 삭제보다 낫다”고 말한다. 공개 범위도 사내·특정 부서·협력사·고객·전체 공개로 나누고 예를 들어 검증된 문서만 고객에게 보이게 한다. [D5]

**티켓을 지식 그래프로 바꾼 예.** LinkedIn은 과거 Jira 티켓을 요약·설명·우선순위·재현 단계 같은 구간으로 나눴다. 정해진 칸은 규칙으로 뽑고 나머지 글은 LLM으로 정리했다. 티켓끼리는 두 가지로 이었다. Jira에 적힌 복제·관련·원인 관계는 명시적 연결로, 제목 임베딩이 비슷한 티켓은 암묵적 연결로 삼았다. 각 구간의 글은 임베딩해 벡터 DB에 넣었다. [D2]

### 7.2 구조화: 분류어, 담당자, 검토일

**분류 체계와 자유 태그.** SharePoint는 관리자가 계층으로 짠 공식 용어 집합(택소노미)과 사용자가 자유롭게 붙이는 키워드(폭소노미)를 함께 쓸 수 있게 한다(검색 요약으로 확인). [D6] 손으로 태그를 붙이기 어렵다 보니 AI가 회사 용어집에서 골라 자동으로 붙여 주는 기능도 나왔다. 다만 지원 형식이 doc·docx·pdf·pptx로 제한되고, 스캔한 PDF는 안 되며, 태그가 붙기까지 20분에서 24시간이 걸린다. 기존 문서는 기본적으로 처리하지 않는다. [D7]

메타데이터는 이제 AI 답변 품질에도 직접 영향을 준다. Microsoft 365 Copilot은 SharePoint 문서함이나 폴더 범위로 질문하면 그 문서함의 열 메타데이터를 관련도 신호로 쓴다. [D8]

**사람과 활동까지 연결하는 지식 그래프.** Glean은 지식 그래프를 세 축으로 설명한다. 문서·메시지·티켓 같은 **콘텐츠**, 여러 시스템에 흩어진 한 사람의 역할·팀·근속 같은 **사람**, 메신저·메일에서 나오는 **활동**이다. [D10] 누가 쓴 문서를 누가 자주 보는지까지 검색 순위에 반영하는 구조다.

**담당자와 검토 주기.** Guru는 카드마다 검증 담당자와 검토 주기(매주~매년)를 정한다. 기한이 지나면 '미검증'으로 바뀌지만 검색에는 남는다. 미검증이면서 쓰이지도 않는 카드는 자동 보관 대기열로 간다. [D11] Glean도 검증한 사람과 날짜를 배지로 보여 준다. [D12] Confluence는 일정 기간 조회·댓글·수정이 없는 페이지를 골라 한꺼번에 보관할 수 있고 기본 기준은 180일이다. [24][25]

### 7.3 색인: AI가 읽을 수 있게 만드는 과정

**커넥터.** Microsoft 365 Copilot 커넥터는 두 종류다. 데이터를 Microsoft Graph에 색인해 두는 **동기화형**과, 색인하지 않고 필요할 때 MCP로 실시간 조회하는 **연합형**이다. 가져온 항목마다 내용·메타데이터·접근 권한 목록(ACL)이 함께 들어가고 새 문서·수정·삭제를 계속 반영한다. 기본 제공 커넥터는 100개가 넘는다. [D13] 색인도 자동이다. 두 명 이상에게 공유된 새 SharePoint 문서는 매일 색인되고 이미 색인된 문서의 수정은 바로 반영된다. [D8]

**청킹.** 긴 문서는 조각으로 잘라 넣는다. Azure AI Search는 512토큰 조각에 25%(128토큰)를 겹치게 하는 설정을 출발점으로 권한다. 문서 중간 조각에 문서 제목을 붙이라는 조언도 있다. [D14]

**하이브리드 검색.** 키워드 검색(BM25)과 의미 검색(벡터)을 함께 돌리고 결과를 합친다. Microsoft 문서는 제품 코드·전문 용어·날짜·사람 이름에는 키워드 검색이 더 잘 맞는다고 설명한다. [D15] 사내 문서에는 품번·조항 번호·사람 이름이 많아서 의미 검색만으로는 부족하다.

**한국만의 숙제, HWP.** 정부·공공 문서의 상당수가 HWP다. 2025년 12월 대통령 업무보고에서 “정부 공문서는 데이터 측면에서 가장 양질의 자산인데 기계가 못 읽는다”는 지적이 나왔고 행정직 14,208명 조사에서 91.1%가 주로 HWP나 PDF를 쓴다고 답했다. [D16] 정부는 2026년 4월 HWP를 XML 기반인 HWPX로 바꾸는 계획을 냈다. 온나라 시스템은 5월 18일부터 지자체에도 HWPX를 의무화했고 공직자통합메일은 10월부터 HWP 첨부를 제한한다. 이유는 HWP의 닫힌 구조 때문에 AI가 내용을 분석하기 어렵다는 것이다. [D17] 한컴이 파는 데이터 로더는 HWP·HWPX를 PDF로 바꾸지 않고 원본에서 바로 표·이미지·제목 구조를 뽑는다. [D18]

### 7.4 권한: AI 검색의 성패가 갈리는 곳

원칙은 간단하다. **원본 시스템에서 못 보는 문서는 AI 답변에도 나오면 안 된다.** 다만 실제로 구현하기는 까다롭다.

- Azure AI Search는 질문하는 사용자의 로그인 정보와 색인에 저장된 권한 값을 비교해 결과를 거른다. 원본 권한이 바뀌어도 다음 동기화 전까지는 반영되지 않는다. 문서를 조각으로 자르면 **모든 조각에 권한 값을 복사해 넣어야** 하고 빠뜨리면 조각 단위 결과가 걸러지지 않는다. [D19]
- Amazon Q Business는 문서마다 사용자 메일·그룹 권한을 색인한다. 권한 변경은 다음 수집 때 반영되므로 자주 다시 동기화하라고 권한다. 한 번의 동기화가 지울 수 있는 문서 비율에 상한을 거는 안전장치도 있다. [D9]
- Glean 개발 문서는 “그룹을 사용자 목록으로 풀어 버리지 말 것”, “민감한 콘텐츠를 '일단' 전체 공개 권한으로 색인하지 말 것”이라고 경고한다. [D20]

코레일이 부서별로 데이터를 나누고 인사정보와 연동해 인증한 것도 같은 문제에 대한 답이다. [12]

### 7.5 정리: 안 쓰는 데이터를 걷어 내기

회사 데이터에는 쓸모없는 것이 많다. Veritas의 2016년 조사는 저장 데이터의 52%가 가치를 알 수 없는 데이터, 33%가 중복·낡음·사소함(ROT) 데이터, 업무에 꼭 필요한 데이터는 15%뿐이라고 추정했다. [D21] 10년 된 수치지만 AI 검색 시대에는 이 문제가 더 커졌다. Microsoft는 생성형 AI가 낡았거나 권한이 과하게 열린 콘텐츠를 끌어올려 과잉 공유 문제를 키운다고 설명한다. [D22]

그래서 Microsoft Purview는 사용량 상위 100개 SharePoint 사이트를 매주 점검하고 몇 가지 조치를 제안한다. 특정 민감도 레이블이 붙은 문서는 Copilot이 요약하지 못하게 하는 DLP 정책, 특정 사이트를 전사 검색과 Copilot에서 빼는 제한 기능, 자동 레이블, 3년 이상 열지 않은 문서를 지우는 보존 정책이다. [D22] 사이트를 검색에서 빼는 기능은 권한 자체를 바꾸지 않으며 Microsoft도 이를 “임시 통제 수단”이라고 부른다. [D23] 그러니 근본 해결은 권한 정리다.

### 7.6 측정: 쌓은 지식이 쓰이는가

KCS는 지식 문서를 몇 건 만들었는지보다 **다시 쓰였는지**를 본다. 참여율, 남의 문서 재사용, 내 문서가 남에게 인용된 횟수를 재고 셀프서비스 성공은 “도움말을 본 뒤 일정 기간 안에 상담을 다시 접수했는가”로 판단한다. [D24] 업계 평균 셀프서비스 성공률은 40~50%, 성숙한 조직은 80~85%라고 KCS 가이드는 적고 있다. [D25]

AI 답변 품질은 따로 잰다. Microsoft Foundry의 RAG 평가 도구는 검색 정확도, 근거에 충실한지(그라운디드니스), 빠뜨린 내용이 없는지(완전성), 질문과 맞는지(관련성)를 1~5점으로 매긴다. 조각 크기·검색 개수·검색 방식을 바꿔 가며 비교하라고 권한다. [D26] Morgan Stanley도 도입 전에 평가 세트부터 만들었다. [D1]

## 8. 지식의 내재화: 문서가 사람의 실력이 되기까지

데이터를 잘 쌓아도 사람이 그 지식을 체득하지 못하면 KMS는 창고에 그친다. “지식의 내재화”를 이론과 실제 사례로 나눠 보자.

### 8.1 이론: 말로 다 할 수 없는 지식

철학자 마이클 폴라니는 “우리는 말할 수 있는 것보다 더 많이 안다”고 했다. [I4] 자전거 타는 법이나 숙련공의 손감각처럼 몸에 밴 지식을 **암묵지**라고 한다. 문서·매뉴얼처럼 말과 글로 옮긴 지식은 **형식지**다.

노나카 이쿠지로는 기업에서 지식이 두 형태를 오가며 커진다고 보고 이를 네 단계로 정리했다. 단계 이름의 머리글자를 따 SECI 모델이라고 부른다. [I5][I6]

| 단계 | 무엇이 일어나나 | 회사에서의 모습 |
|---|---|---|
| 공동화(Socialization) | 암묵지 → 암묵지 | 선배 옆에서 일하며 감을 익힌다 |
| 표출화(Externalization) | 암묵지 → 형식지 | 감을 말과 글, 그림으로 옮긴다 |
| 연결화(Combination) | 형식지 → 형식지 | 여러 문서를 묶어 매뉴얼·시스템으로 만든다 |
| 내재화(Internalization) | 형식지 → 암묵지 | 매뉴얼대로 직접 해 보며 다시 내 감각으로 만든다 |

노나카는 내재화를 “형식지를 암묵지로 체화하는 과정”이라고 정의하고 **해 보면서 배우기**와 가깝다고 했다. 형식지는 행동과 실습으로 실현돼야 한다는 것이다. 예로 교육 프로그램, 매뉴얼을 보며 돌아보기, 시뮬레이션과 실험을 든다. [I6]

유명한 예가 1985년 마쓰시타의 제빵기 개발이다. 소프트웨어 개발자 다나카 이쿠코는 오사카의 한 호텔 제빵장에게 직접 배우며 반죽을 늘이는 독특한 손놀림을 발견했다(공동화). 1년의 시행착오 끝에 엔지니어들과 그 동작을 사양으로 옮겼고(표출화), 기계 안에 특수한 돌기를 넣은 제품으로 만들었다(연결화). 제품은 신규 주방가전 첫해 판매 기록을 세웠다. 이 경험을 거치며 개발팀은 자기 암묵지를 넓혔다(내재화). 노나카는 표출화와 내재화가 이 순환에서 가장 중요한 단계이며 둘 다 개인의 적극적인 참여가 필요하다고 강조했다. [I5]

노나카는 지식이 생기는 **장소**도 중요하게 봤다. 일본어로 '바(ba)'라고 하는 이 개념은 물리적·가상·정신적 공간을 모두 포함한다. [I7] 내재화가 일어나는 곳은 '실천의 장'이다. 세븐일레븐 재팬 점원들이 매일 판매 데이터로 자기 가설을 확인하는 매장이 그 예다. [I6]

### 8.2 실제로 하는 일

**사후 검토(AAR).** 미 육군에서 시작된 방법으로, 일이 끝나면 “무엇을 하려 했나, 실제로 무슨 일이 있었나, 왜 그랬나, 다음엔 무엇을 지키고 무엇을 고칠까”를 함께 이야기한다. 미 육군은 2025년 2월 AAR 교범 TC 7-0.1을 새로 냈는데 마지막 장이 후속 조치(재훈련, 표준 절차 수정, 교훈 공유)다. [I8] 내재화는 돌아본 내용이 다음 행동을 바꿀 때 이루어진다.

**실천 공동체.** 같은 일을 하는 사람들이 부서를 넘어 모이는 모임이다. 세계은행은 1996년 '지식 은행'이 되겠다고 선언했는데, 내부 자료는 사람을 잇는 일도 문서를 모으는 일도 실천 공동체 없이는 효과가 없었고, 주제별 그룹이 “가장 가치 있는 KM 산출물”이 됐다고 평가했다. [I9] 석유 서비스 기업 슐럼버거의 Eureka 공동체는 1999년 기사에서 “기술이 아니라 사람”으로 소개됐다. [I10]

**은퇴자의 지식 손실 관리.** 숙련자가 한꺼번에 은퇴하는 원자력 업계는 이 문제를 일찍부터 다뤘다. 국제원자력기구(IAEA)는 2017년 원자력 조직의 지식 손실 위험관리 지침을 냈다. 은퇴 시기, 그 자리의 중요도, 그 지식을 얼마나 옮기고 문서로 만들 수 있는지를 따져 위험을 평가하고, 대응 조치를 정하고, 계속 점검하는 구조다. [I11] 미국 원자력규제위원회(NRC)는 2006년 KM 프로그램을 공식화했다. 은퇴한 직원을 다시 고용해 멘토로 쓰고, 신입과 선임을 짝지어 규제 지침 검토를 맡기며, TMI 같은 주요 사건을 NUREG/KM 시리즈로 기록한다. [I12]

**국내 기업.**

- 삼성 지역전문가: 1990년 도입했다. 입사 3년 차 이상 직원이 1~2년간 해외에서 스스로 계획을 짜 언어와 문화를 익힌다. 80개 넘는 나라에서 약 3,500명이 거쳤다(2019년 보도). [I13]
- 포스코 명장: 2015년 도입해 지금까지 30명을 뽑았다. 명장은 기술 전수, 사내 대학 강의, 신입 교육, 협력사 기술 지원을 맡고 은퇴 뒤에도 기술 자문으로 일할 수 있다. 포스코는 이들의 암묵지를 디지털 자산으로 바꿔 AI와 결합하는 작업을 하고 있다. [I14]
- 정부 '온나라 지식'의 업무위키: 공무원 전용 폐쇄형 플랫폼에서 매뉴얼 공유로 시작해 누구나 고칠 수 있는 업무 노하우 공유로 넓혀 왔다. 남해군의 한 간호직 공무원이 직접 만든 문서 서식 도구를 올린 사례가 소개됐다. [I15]
- 사내 대학과 사내 강사: SK하이닉스는 사내 전문가가 만든 약 100개 과정을 운영하고 지난해 7,326명이 들었다. 삼성은 SEU와 DS대학교를, LG는 직원이 강사로 나서는 LG Qurious를 운영한다. [I16]

### 8.3 무엇이 효과가 있었나

**보상이 오히려 역효과를 낼 수 있다.** 국내 27개 조직의 관리자 154명을 조사한 2005년 연구에서, 금전 같은 외적 보상을 기대할수록 지식 공유에 대한 태도가 오히려 나빠졌다. 공유를 이끈 것은 서로 돕는 관계에 대한 기대, 자기 가치감, 조직 분위기였다. [I17] APQC 2026년 조사에서도 KM의 가장 큰 위협은 “지식 공유를 장려하지 않는 조직문화”(40%)였다. [1]

**기술만으로는 안 된다.** 1999년 McDermott는 지식 활용이 정보기술보다 공동체 만들기에 더 좌우된다고 주장했다. [I18] 국제표준 ISO 30401도 “기술 시스템을 사기만 하면 KM이 가치를 낼 것”이라는 생각을 흔한 오해로 꼽는다. [4] 앞서 본 NASA LLIS 사례가 그 증거다.

**업무 흐름 안에 넣어야 한다.** APQC는 KM을 평소 업무에서 빠져나와 따로 하는 '흐름 밖' 활동과 일상 업무에 녹아든 '흐름 안' 활동으로 나눈다. 모범 조직은 단계별 심사(스테이지 게이트) 같은 기존 절차 안에 지식 공유를 넣고 책임 소재를 정해 관리한다. [I19] KCS의 “일하는 도중에 기록한다”와 같은 이야기다.

### 8.4 AI는 내재화를 돕는가, 방해하는가

연구 결과는 양쪽으로 나뉜다.

**돕는다는 쪽.** 고객 상담원 5,179명을 분석한 연구에서 AI 도우미를 쓰자 시간당 문제 해결 건수가 평균 14% 늘었고 신입·저숙련 상담원은 34% 늘었다. 숙련 상담원에게는 효과가 거의 없었다. 연구진은 AI가 잘하는 직원의 노하우를 퍼뜨려 신입이 경험 곡선을 더 빨리 오르게 돕는다는 증거로 해석했다. [I20] 숙련자의 암묵지가 AI를 거쳐 신입에게 옮겨 간다고 볼 수 있다.

**방해한다는 쪽.**

- Microsoft Research와 카네기멜런대가 지식근로자 319명의 실제 업무 936건을 조사했더니 생성형 AI를 믿을수록 비판적 사고를 덜 했다. 자기 실력을 믿는 사람일수록 AI 결과를 더 확인하고 다듬었다. 연구진은 장기적으로 과의존과 독립적 문제 해결 능력의 저하를 경고했다(자기 보고식 설문). [I21]
- Anthropic이 2026년 1월 낸 무작위 실험에서는 주니어 개발자 52명이 새 라이브러리를 배웠다. AI를 쓴 그룹은 이해도 퀴즈에서 50점, 직접 코딩한 그룹은 67점을 받았고 차이는 디버깅에서 가장 컸다. AI 그룹이 빨라진 시간은 2분 정도로 통계적으로 의미가 없었다. 같은 AI 그룹 안에서도 일을 통째로 맡긴 사람들은 평균 40%를 밑돌았고 개념을 묻거나 코드를 받은 뒤 이해하려 한 사람들은 65% 이상이었다. [I22]
- 폴란드 4개 병원의 내시경 전문의 19명은 AI 보조를 도입한 뒤, AI 없이 한 대장내시경에서 선종 발견율이 28.4%에서 22.4%로 떨어졌다(관찰 연구). [I23]

**정리.** 노나카가 말한 내재화의 조건은 “직접 해 보기”와 “개인의 참여”였다. AI가 답을 대신 내주면 그 과정이 빠진다. Anthropic 실험에서는 AI를 질문 상대로 쓴 사람이 덜 잃었다. 그래서 AI를 막을지 정하는 일보다 **어떻게 쓰게 할지 설계하는 일**이 내재화의 새 과제가 됐다.

## 9. 정리: 도입한다면 무엇부터 볼까

아래는 이 글에서 다룬 출처들을 묶어 내린 판단이다.

| 확인할 것 | 왜 | 근거 |
|---|---|---|
| 권한부터 정리했나 | AI 검색은 숨은 과잉 공유를 그대로 드러낸다. 조각마다 권한을 넣어야 한다 | 6.1, 7.4 |
| 문서마다 담당자와 검토일이 있나 | 낡은 문서가 AI 답변의 근거가 된다 | 7.2, 7.5 |
| 기록이 업무 흐름 안에 있나 | 따로 시간을 내야 하는 기록은 쌓이지 않는다 | 7.1, 8.3 |
| 등록 건수 대신 재사용을 재나 | 건수만 늘고 질이 정체된 사례가 있다 | 6.1, 7.6 |
| 도입 전에 평가 세트를 만들었나 | 무엇이 좋아졌는지 잴 수 없다 | 6.1, 7.6 |
| 비용을 좌석과 사용량으로 나눠 봤나 | AI는 상위 요금제 묶음과 크레딧으로 과금된다 | 3, 4 |
| 우리 문서 형식을 읽을 수 있나 | 국내는 HWP·스캔 PDF가 많다 | 7.3 |
| 사람이 배우는 장치가 있나 | 문서와 AI만으로는 실력이 쌓이지 않는다 | 8 |

## 10. 자료를 읽을 때 주의할 점

- **업체 자료가 많다.** Morgan Stanley 수치는 OpenAI 사례 자료이고 CMS 이전 사례는 모두 해당 업체가 낸 것이다. CoreView·Storyblok·Liferay 조사도 관련 제품을 파는 쪽이 냈다.
- **확인 수준이 다르다.** Microsoft Copilot 단독 가격, Contentful 가격, SharePoint 자동 태그와 일부 국내 제품 정보는 공식 페이지를 직접 열지 못하고 검색 요약이나 제3자 자료로 확인했다. Glean 가격은 공개 자료가 없어 적지 않았다.
- **오래된 수치가 있다.** Veritas ROT 수치는 2016년, NASA LLIS 이용률은 2012년, GKMC 평가는 2014년 보도다.
- **바뀌는 중인 것이 많다.** Amazon Q Business는 신규 고객을 받지 않고 Azure의 일부 권한 기능은 미리보기 단계다. 국내 공공 클라우드 인증 제도는 2027년 하반기까지 바뀐다.
- **확인하지 못한 것.** 국내 기업이 사내 문서를 RAG용으로 정제한 공개 사례, 한수원의 은퇴자 지식 전수 프로그램, 이름이 공개된 국내 기업의 헤드리스 CMS 도입 사례는 찾지 못했다.


## 참고 자료

모든 링크는 2026년 10월 14일에 확인했다. 번호 [1]~[29]는 앞선 리서치 문서의 출처 번호를 그대로 썼다.

**앞선 리서치에서 이어 쓴 출처**

- [1] APQC, 2026 Knowledge Management Priorities and Trends Survey Report — https://www.apqc.org/resource-library/resource-listing/2026-knowledge-management-priorities-and-trends-survey-report
- [4] BSI, ISO 30401 Knowledge Management — https://bsigroup.com/en-AE/industries-and-sectors/Government/ISO-30401-Knowledge-Management
- [11] Synthedia, Morgan Stanley Rolls Out OpenAI GPT-Powered Assistant — https://synthedia.substack.com/p/morgan-stanley-rolls-out-openai-gpt
- [12] 뉴스1, 코레일 '에어파인더' 도입 (2026.2) — https://www.news1.kr/amp/local/daejeon-chungnam/6077390
- [13] LG, LG CNS 'AI를 활용한 KM 혁신' 서비스 개시 — https://www.lg.co.kr/media/release/26907
- [15] 헬로티, 코오롱베니트 '코아이봇' 도입 (2024.8) — https://www.hellot.net/mobile/article.html?no=92273
- [17] Computerworld, Microsoft 365 Copilot rollouts slowed by data security, ROI concerns — https://www.computerworld.com/article/3542000/microsoft-365-copilot-rollouts-slowed-by-data-security-roi-concerns.html
- [19] Infosecurity Magazine, Microsoft Copilot Deployments Delayed Over Security Concerns — https://infosecurity-magazine.com/news/microsoft-copilot-delayed-over
- [20] Arabian Post, Security gaps stall Microsoft Copilot deployments — https://thearabianpost.com/security-gaps-stall-microsoft-copilot-deployments/
- [21] W3Techs, Market share trends for content management systems — https://w3techs.com/technologies/history_overview/content_Management
- [22] W3Techs, Usage statistics of content management systems — https://w3techs.com/technologies/overview/content_management
- [24] Atlassian, Introducing page and bulk archiving in Confluence Cloud — https://www.atlassian.com/blog/confluence/introducing-page-and-bulk-archiving-in-confluence-cloud
- [25] Atlassian Community, Manage your stale pages in bulk — https://answers.atlassian.com/t5/Confluence-Cloud-Admins-articles/Manage-your-stale-pages-in-bulk/ba-p/2592287
- [27] Contentful Help Center, Create a custom role — https://www.contentful.com/help/creating-a-custom-role/
- [28] Contentful, Content Management API – Roles — https://www.contentful.com/developers/docs/references/content-management-api/roles.md
- [29] 서울신문, '지식행정' 활성화 공직 경쟁력 높인다 (2014.5) — https://m.go.seoul.co.kr/news/2014/05/01/20140501019026

**KMS 제품·판매 [K]**

- [K1] Atlassian, Confluence pricing — https://www.atlassian.com/software/confluence/pricing
- [K2] Atlassian, Rovo licensing — https://www.atlassian.com/licensing/rovo
- [K3] Microsoft Learn, Microsoft 365 Copilot licensing — https://learn.microsoft.com/en-us/copilot/microsoft-365/microsoft-365-copilot-licensing
- [K4] Microsoft, Microsoft 365 E7 — https://www.microsoft.com/en-us/microsoft-365/enterprise/e7 · Directions on Microsoft — https://www.directionsonmicrosoft.com/m365-e7-to-launch-may-1-for-99-per-user-per-month/
- [K5] Office Watch, Microsoft 365 E7 (제3자 자료) — https://office-watch.com/2026/microsoft-365-e7-a-99-frontier-suite-the-future-or-an-expensive-bundle/
- [K6] Microsoft Learn, SharePoint Advanced Management features with Copilot license — https://learn.microsoft.com/en-my/sharePoint/sharepoint-advanced-management-features-copilot-license
- [K7] Notion, 요금제 — https://www.notion.com/pricing
- [K8] Glean, Connectors — https://www.glean.com/connectors
- [K9] Guru, Pricing — https://www.getguru.com/pricing
- [K10] 9to5Google, Google launches Gemini Enterprise (2025.10.9) — https://9to5google.com/2025/10/09/google-gemini-enterprise/
- [K11] AWS, Amazon Q Business pricing — https://aws.amazon.com/q/business/pricing/
- [K12] ServiceNow, ServiceNow completes acquisition of Moveworks — https://newsroom.servicenow.com/press-releases/details/2025/ServiceNow-completes-acquisition-of-Moveworks/default.aspx
- [K13] Salesforce Ben, ServiceNow complete record-breaking deal to buy Moveworks — https://www.salesforceben.com/servicenow-complete-record-breaking-deal-to-buy-ai-platform-moveworks-for-2-85b/
- [K14] Zendesk, 요금제 — https://www.zendesk.kr/pricing/
- [K15] Salesforce, Service Cloud editions and pricing — https://www.salesforce.com/editions-pricing/service-cloud/
- [K16] Consortium for Service Innovation, KCS v6 Verified tool providers — https://www.serviceinnovation.org/kcs-v6-verified-tool-providers-recognized-with-digital-badges/
- [K17] 아이뉴스24, 코난테크놀로지 법제처 생성형 AI 법령검색 사업 수주 (2026.6) — https://www.inews24.com/view/1973190
- [K18] 바이라인네트워크, 솔트룩스 중앙노동위원회 사업 수주 (2025.7) — https://byline.network/2025/07/73194/
- [K19] 이코노미스트, 솔트룩스 LUXIA-ON (2026.3) — https://economist.co.kr/article/view/ecn202603170029
- [K20] 바이라인네트워크, 와이즈넛 사업 구조 (2025.1) — https://byline.network/2025/01/8-303/
- [K21] 보안뉴스, 사이냅소프트 도큐애널라이저 — https://m.boannews.com/html/detail.html?idx=128415
- [K22] 디지털데일리, 범정부 AI 공통기반 사업 (2025.11) — https://www.ddaily.co.kr/page/view/2025111215142078532
- [K23] 아이뉴스24, 범정부 AI 공통기반 시범 운영 (2026.10) — https://inews24.com/view/2013271
- [K24] ITWorld, 네이버클라우드 뉴로클라우드 for HyperCLOVA X — https://www.itworld.co.kr/news/311946
- [K25] 바이라인네트워크, LG CNS 2026년 1분기 실적 — https://byline.network/2026/04/30-525/
- [K26] 바이라인네트워크, 국가 망보안체계(N²SF) 초안 (2025.1) — https://byline.network/2025/01/30-371/
- [K27] 머니투데이방송, 공공 클라우드 검증 국정원 일원화 (2026.4) — https://news.mtn.co.kr/news-detail/2026042012075784201
- [K28] 바이라인네트워크, 공개 등급 상용 생성형 AI 검증 면제 방침 (2026.9) — https://byline.network/2026/09/18-635/
- [K29] 아이뉴스24, 소프트웨어 유지보수 요율 (2025.9) — https://www.inews24.com/view/1885259

**CMS 제품·운영 [C]**

- [C1] Bynder, What is digital asset management — https://www.bynder.com/en/what-is-digital-asset-management/
- [C2] W3Techs, Usage statistics of content management systems (2026.10.14) — https://w3techs.com/technologies/overview/content_management
- [C3] WordPress.org, License — https://wordpress.org/about/license/
- [C4] WordPress VIP, Pricing — https://wpvip.com/pricing/
- [C5] Shopify, Pricing — https://www.shopify.com/pricing
- [C6] Wix, Plans — https://www.wix.com/plans
- [C7] Squarespace, Pricing — https://www.squarespace.com/pricing
- [C8] Contentful, Pricing · Usage limits (검색 요약) — https://www.contentful.com/pricing/ · https://www.contentful.com/help/usage-limits-january-2026/
- [C9] Sanity, Pricing — https://www.sanity.io/pricing
- [C10] Strapi, Cloud pricing — https://strapi.io/pricing-cloud
- [C11] Strapi Docs, Review Workflows — https://docs.strapi.io/cms/features/review-workflows
- [C12] Storyblok, Pricing — https://www.storyblok.com/pricing
- [C13] Optimizely, Pricing — https://www.optimizely.com/pricing/
- [C14] Adobe Experience League, AI agents in AEM — https://experienceleague.adobe.com/en/docs/experience-manager-cloud-service/content/ai-in-aem/agents/overview
- [C15] Gilbane, Contentful launches new CMS capabilities and Shopify partnership (2025.3) — https://gilbane.com/2025/03/contentful-launches-new-cms-capabilities-and-shopify-partnership/
- [C16] Adobe, General availability of AI agents (2025.9.10) — https://news.adobe.com/news/downloads/pdfs/2025/09/09-10-2025-adobe-announces-general-availability-of-ai-agents.pdf
- [C17] Sitecore Developers, SitecoreAI FAQ — https://developers.sitecore.com/sitecoreai/faq
- [C18] WordPress.org, WordPress 7.0 “Armstrong” (2026.5.20) — https://wordpress.org/news/2026/05/armstrong/
- [C19] Adobe Experience League, AEM as a Cloud Service architecture — https://experienceleague.adobe.com/en/docs/experience-manager-cloud-service/content/overview/architecture
- [C20] Samsung SDS, AEM migration case study — https://www.samsungsds.com/en/case-study/aem-migration.html
- [C21] Storyblok, Marc O'Polo case study — https://www.storyblok.com/cs/marc-o-polo
- [C22] Contentful, Align case study (검색 요약) — https://www.contentful.com/case-studies/align/
- [C23] GitHub, gnuboard5 — https://github.com/gnuboard/gnuboard5
- [C24] GitHub, rhymix — https://github.com/rhymix/rhymix
- [C25] 코민, SiiRU CMS 소개 — https://docs.comin.com/siiru-cms/v2.1/intro
- [C26] 디지털데일리, 디지털포용법 시행 (2026.1) — https://www.ddaily.co.kr/page/view/2026012110374732032
- [C27] 문화체육관광부, 웹 접근성 품질마크 — https://www.mcst.go.kr/kor/s_etc/webAccess/accessibility.jsp
- [C28] KED Global, LG CNS–Adobe Korea MOU (2023.2) — https://www.kedglobal.com/newsView/ked202302230024

**데이터 수집·구조화·색인·측정 [D]**

- [D1] OpenAI, AI in the Enterprise — https://cdn.openai.com/business-guides-and-resources/ai-in-the-enterprise.pdf
- [D2] Xu et al., Retrieval-Augmented Generation with Knowledge Graphs for Customer Service Question Answering (SIGIR 2024) — https://arxiv.org/abs/2404.17723
- [D3] KCS v6 Practices Guide, Capture in the workflow — https://library.serviceinnovation.org/KCS/KCS_v6/KCS_v6_Practices_Guide/030/030/010/010
- [D4] KCS v6 Practices Guide, Article structure — https://library.serviceinnovation.org/KCS/KCS_v6/KCS_v6_Practices_Guide/030/040/010/020
- [D5] KCS v6 Practices Guide, Article state and visibility — https://library.serviceinnovation.org/KCS/KCS_v6/KCS_v6_Practices_Guide/030/040/010/030
- [D6] Microsoft Learn, Managed metadata (검색 요약) — https://learn.microsoft.com/en-us/sharepoint/managed-metadata
- [D7] Microsoft Learn, Taxonomy tagging overview — https://learn.microsoft.com/en-us/microsoft-365/documentprocessing/taxonomy-tagging-overview
- [D8] Microsoft Learn, Semantic indexing for Microsoft 365 Copilot — https://learn.microsoft.com/en-us/microsoftsearch/semantic-index-for-copilot
- [D9] AWS, Amazon Q Business connector concepts — https://docs.aws.amazon.com/amazonq/latest/qbusiness-ug/connector-concepts.html
- [D10] Glean Docs, Knowledge graph — https://docs.glean.com/security/knowledge-graph
- [D11] Guru, Verification — https://www.getguru.com/features/verification
- [D12] Glean Docs, How verification works — https://docs.glean.com/user-guide/knowledge/verification/how-verification-works
- [D13] Microsoft Learn, Microsoft 365 Copilot connectors overview — https://learn.microsoft.com/en-us/microsoft-365/copilot/connectors/overview
- [D14] Microsoft Learn, Chunk documents for vector search — https://learn.microsoft.com/en-us/azure/search/vector-search-how-to-chunk-documents
- [D15] Microsoft Learn, Hybrid search overview — https://learn.microsoft.com/en-us/azure/search/hybrid-search-overview
- [D16] 다음뉴스, 정부 공문서 HWP 문제 (2025.12) — https://v.daum.net/v/20251218060302375
- [D17] 바이라인네트워크, 정부 HWP→HWPX 전환 계획 (2026.4) — https://byline.network/2026/04/23-586/
- [D18] 한컴, Hancom Data Loader — https://www.hancom.com/en/ai/products/dataloader
- [D19] Microsoft Learn, Document-level access control in Azure AI Search — https://learn.microsoft.com/en-us/azure/search/search-document-level-access-overview
- [D20] Glean Developers, Indexing SDK permissions — https://developers.glean.com/libraries/indexing-sdk/permissions
- [D21] Veritas, Global Databerg Report (2016, 검색 요약) — https://www.veritas.com/news-releases/2016-03-15-veritas-global-databerg-report-finds-85-percent-of-stored-data
- [D22] Microsoft Learn, Data Security Posture Management for AI — https://learn.microsoft.com/en-us/purview/dspm-for-ai
- [D23] Microsoft Learn, Restricted Content Discovery — https://learn.microsoft.com/en-us/sharepoint/restricted-content-discovery
- [D24] KCS v6 Practices Guide, Measurement — https://library.serviceinnovation.org/KCS/KCS_v6/KCS_v6_Practices_Guide/070
- [D25] KCS v6 Practices Guide, Self-service — https://library.serviceinnovation.org/KCS/KCS_v6/KCS_v6_Practices_Guide/030/040/010/080
- [D26] Microsoft Learn, RAG evaluators — https://learn.microsoft.com/en-us/azure/foundry/concepts/evaluation-evaluators/rag-evaluators

**지식의 내재화 [I]**

- [I1] 동아일보(다음뉴스), 한수원 생성형 AI '하이누리' (2026.5) — https://v.daum.net/v/20260528043351848
- [I2] NASA APPEL, Lessons learned lifecycle — https://appel.nasa.gov/lessons-learned/lessons-learned-lifecycle-and-highlights/
- [I3] NASA OIG, IG-12-012 (2012) — https://www.oversight.gov/sites/default/files/oig-reports/IG-12-012.pdf · SpacePolicyOnline 요약 — https://spacepolicyonline.com/news/nasa-ig-lessons-learned-system-rarely-used-marginalized/
- [I4] Wikipedia, Tacit knowledge — https://en.wikipedia.org/wiki/Tacit_knowledge
- [I5] Nonaka, The Knowledge-Creating Company (HBR) — https://lumsa.it/sites/default/files/UTENTI/u95/LM51_ITA_The%20Knowledge-Creating%20Company.pdf
- [I6] Nonaka, Toyama & Konno, SECI, Ba and Leadership (Long Range Planning, 2000) — https://tingene.tekniskmuseum.no/images/PDF/Litteratur_MK/SECI_Ba_and_Leadership_a_Unified_Model.pdf
- [I7] Nonaka & Konno, The Concept of “Ba” (California Management Review, 1998) — https://cmr.berkeley.edu/1998/05/40-3-the-concept-of-ba-building-a-foundation-for-knowledge-creation
- [I8] U.S. Army, TC 7-0.1 After Action Reviews (2025.2) — https://www.army.mil/article/283214/tmd_publishes_training_circular_to_augment_fm_and_adp_7_0
- [I9] World Bank, Knowledge management at the World Bank — https://documents.worldbank.org/en/publication/documents-reports/documentdetail/574241468763497022
- [I10] Oil IT Journal, Schlumberger Eureka (1999) — https://oilit.com/HTML_Articles/1999_10_8.php
- [I11] IAEA, Knowledge Loss Risk Management in Nuclear Organizations (NG-T-6.11, 2017) — https://www.iaea.org/node/41993
- [I12] U.S. NRC, NUREG/KM knowledge management series — https://www.nrc.gov/reading-rm/doc-collections/nuregs/knowledge/index
- [I13] 뉴시스, 삼성 지역전문가 제도 (2019.11) — https://mobile.newsis.com/view/NISX20191108_0000824118
- [I14] 파이낸셜뉴스, 포스코 명장 (2026.7) — https://www.fnnews.com/news/202607151332134058
- [I15] 경향신문, 온나라 지식 업무위키 (2023.5) — https://www.khan.co.kr/article/202305101200001
- [I16] 청년일보, 기업 사내 교육 (2026.7) — https://www.youthdaily.co.kr/news/article.html?no=223092
- [I17] Bock, Zmud, Kim & Lee, MIS Quarterly 29(1), 2005 — https://aisel.aisnet.org/misq/vol29/iss1/6
- [I18] McDermott, Why Information Technology Inspired But Cannot Deliver Knowledge Management (CMR, 1999) — https://cmr.berkeley.edu/1999/08/41-4-why-information-technology-inspired-but-cannot-deliver-knowledge-management
- [I19] APQC, Embedding knowledge management in business processes and workflows — https://www.apqc.org/blog/embedding-knowledge-management-business-processes-and-workflows
- [I20] Brynjolfsson, Li & Raymond, Generative AI at Work (NBER w31161) — https://www.nber.org/papers/w31161
- [I21] Lee et al., The Impact of Generative AI on Critical Thinking (CHI 2025) — https://advait.org/files/lee_2025_ai_critical_thinking_survey.pdf
- [I22] Anthropic, How AI assistance impacts the formation of coding skills (2026.1.29) — https://www.anthropic.com/news/AI-assistance-coding-skills
- [I23] STAT, AI deskilling in colonoscopy (Lancet Gastroenterology & Hepatology, 2025) — https://www.statnews.com/2025/08/12/ai-deskilling-doctors-colonoscopy-study-lancet/

<!-- LEARNING-END -->
---

## 읽은 뒤 확인하고 다음으로

**확인 질문:** 지식이 기록·검색·체득되는 과정과 그 비용 구조 이해하기를 목표로, 한 회사가 장애 처리 기록을 AI 검색에 올린다고 할 때 무엇을 먼저 정리해야 하는지 자기 말로 설명해 보자. 문서를 많이 쌓는 것과 직원의 실력이 느는 것이 왜 다른지도 덧붙여 보자.

이 경로의 마지막 글이다. [학습 순서]({{< relref "learn" >}})로 돌아가 다른 경로를 고르거나, 사내 문서 검색의 원리가 궁금하면 [RAG 전체 흐름]({{< relref "posts/rag-survey-2026" >}})을 읽어 보자.

[전체 학습 순서]({{< relref "learn" >}}) · [용어 사전]({{< relref "ai-glossary" >}})

<!-- LEARNING-END-END -->
