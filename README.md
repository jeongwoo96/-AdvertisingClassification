# AdvertisingClassification

Fisrt mini project

텍스트마이닝 & 을 이용해 블로그 광고글 구분하기

---

### 프로젝트 목적
- 많은 블로그 체험 후기글이 사실은 돈을 받고 쓴 광고글이고 이로 인해 블로그 글에 대한 신뢰성이 하락함.
- 텍스트 마이닝과 ML을 통해 광고글을 구분하고 블로그 글에 대한 신뢰성을 회복시키고 소비자의 현명한 소비를 도움.
##
### 프로젝트 기간
- 22.07.13 ~ 22.08.04
##
### 광고글과 비광고글의 경향 파악

<img src="https://user-images.githubusercontent.com/112039781/211185026-5d718b0a-ebcd-4e5b-accc-29df26f86902.png">

<img src="https://user-images.githubusercontent.com/112039781/211185061-4ee32f04-b728-4f76-aa1c-0fb01046646f.png">

- 해당 패턴을 바탕으로 수집한 데이터의 라벨링 작업
#
### 데이터 수집

크롤링 실행 : DataGet.ipynb

**Step1**. 크롤링 클래스 객체 선언
```
food = blog_restaurants()
```
　　
```
지역을 입력해주세요 : 부산 해운대
```
　　
  
**Step2**. 키워드에 해당하는 지역 맛집 게시글 크롤링
- ex) '부산 해운대 맛집' 키워드로 작성된 블로그 글 크롤링
```
food.blog_restaurant_get()
```
  
<img src="https://user-images.githubusercontent.com/112039781/211186125-fbb703de-3298-462f-a4e0-f69e7960d5f5.png">

　　
  
**Step3**. 가져온 게시글에 해당하는 식당명 크롤링 
```
food.naver_restaurants_get()
```

　　
  
**Step4**. 글이 많이 게시된 식당 10개 선별
```
food.top_10_restaurant_get()
```

　　
  
**Step5**. 10개의 식당에 대한 글 크롤링 후 저장
```
food.restaurant_get()
```
##

### 크롤링 결과
<img src="https://user-images.githubusercontent.com/112039781/211186314-b051bd7f-e037-491a-b62a-6fa314d40565.png">

##

### 전처리
- Null값 확인 및 제거
```
data.dropna(inplace=True)
```
- 한글 형태소 분석기(Okt)를 활용해 형태소 토큰화 및 품사 태깅
- 불용어 사전 작성 및 제거
([불용어 사전](https://github.com/jeongwoo96/-AdvertisingClassification/blob/main/nsmc_stopwords_5%EC%B0%A8.txt))
-정규 표현식을 활용해 한글 및 공백을 제외한 문자 제거
- 광고글 분석에 유의미한 명사, 동사, 형용사 를 제외한 품사 제거
- StandardScaler를 활용한 본문을 제외한 나머지 특성 (이미지 수(Img), 비디오 유무(Video), 지도 유무(Map), ...) 표준화 작업
