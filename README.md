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
+ Null 값 및 중복값 확인 및 제거
```
data.dropna(inplace=True)
data = data.drop_duplicates(['body'],keep='first')
```
+ 정규 표현식을 활용해 한글 및 공백을 제외한 문자 제거
+ 한글 형태소 분석기인 Okt를 활용해 형태소 토큰화 및 품사태깅
+ 불용어사전 작성 및 불용어 제거(불용어사전 : ['nsmc_stopwords_5차.txt'](https://github.com/Yeons2013/AdvertisingClassification/blob/master/nsmc_stopwords_5%EC%B0%A8.txt))
+ 광고 글 구분에 유의미한 명사, 형용사, 동사를 제외한 품사 제거
```
def preprocessing(review):
    okt = Okt()
    
    # 불용어 22,678개
    f = open('nsmc_stopwords_5차.txt')
    stop_words = f.read().split()
    
    # 1. 한글 및 공백을 제외한 문자 모두 제거. → 한글 이외의 문자는 광고 구분에 중요하지 않다고 판단
    review_text = re.sub("[^가-힣\\s]", "", review)
    
    # 2. okt 객체를 활용해서 형태소 토큰화 + 품사 태깅
    word_review = okt.pos(review_text, stem=True)

    # 3. 노이즈 & 불용어 제거 → 광고 구분에 유의미한 토큰을만을 선별하기 위한 작업
    word_review = [(token, pos) for token, pos in word_review if not token in stop_words and len(token) > 1]
 
    # 4. 명사, 동사, 형용사 추출
    word_review = [token for token, pos in word_review if pos in ['Noun', 'Verb', 'Adjective']]
    
    return word_review
```
+ Standard Scaler를 활용해 본문을 제외한 나머지 (ex>이미지개수, 스티커수, 문장길이..) 특성 표준화 작업
```
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(train_input2)

train_scaled2 = ss.transform(train_input2)
test_scaled2 = ss.transform(test_input2)
```
+ TF-IDF Vectorizer를 활용한 본문 벡터화

##

### ML 모델 학습

#### 본문 내용을 제외한 특성으로 학습
<img src="https://user-images.githubusercontent.com/112039781/211187590-2a447b1a-8150-4748-9684-4dce5818fb55.png">

### ※ 사용 모델

**① RandomForestClassifier**

◆ 선정이유
+ 과대적합 문제 최소화하여 모델의 정확도 향상
+ 대용량 데이터 처리에 효과적
+ Classification 및 Regression 문제에 모두 사용 가능

**② ExtraTreesClassifier**

◆ 선정이유
+  feature중에 아무거나 고른 다음 그 feature에 대해 최적의 Node를 분할
+ 준수한 성능을 보이며 과대적합을 막고 검증 세트의 점수를 높이는 효과가 있음.
+ 속도가 빠르다는 장점이 있음.
<img src="https://media.discordapp.net/attachments/1022477080031666276/1061568202531934268/image.png">


#### 본문 내용만을 이용한 학습

- 본문 내용을 제외한 특성들은 특정 토큰으로 치환
<img src="https://user-images.githubusercontent.com/112039781/211188300-3797c0d6-ccec-49d8-8b36-95aa1660063d.png">

#### ※ 사용 모델


RandomForestClassifier, ExtraTreesClassifier이외에 추가로 2개 더 선정

**③ LogisticRegression**

◆ 선정이유
+ 로지스틱 회귀는 매우 효율적이고 엄청난 양의 계산 리소스를 필요로 하지 않기 때문에 널리 사용됨.
+ 쉽게 구현되고 학습하기 쉬우므로 다른 복잡한 알고리즘의 성능을 측정하는 데 도움이 되는 훌륭한 기준이 됨.

**④ *AdaBoostClassifier**

◆ 선정이유
+ AdaBoost는 구현하기 쉬움.
+ 약한 분류기의 실수를 반복적으로 수정하고 약한 학습자를 결합하여 정확도를 높임.
+ andomForest와 비교하였을 때 대체로 boosting이 속도가 더 빠르고 결과가 더 좋게 나옴.
<img src="https://media.discordapp.net/attachments/1022477080031666276/1061574330611413072/image.png">
