# **Лабораторная работа по теме "Изучение эффективности методов тематического моделирования на больших корпусах текста".**

### **Цель проекта:**
Разработка и обучение нейронной сети для автоматического сжатия больших объемов текста, преобразуя их в краткие, содержательные резюме.
Проект реализован с учетом специфики русскоязычных текстов и ориентирован на обработку новостных статей от российских СМИ. Данный инструмент может быть полезен для быстрого получения сжатых версий новостных статей, что упрощает их анализ и восприятие, но так же может подходить и для других видов текста.

### **Исходные данные:**
Использование русскоязычной базы данных новостных статей от российских СМИ. Этот корпус текста послужил основой для дообучения модели T5ForConditionalGeneration.

### **Этапы реализации:**

1. Выбор модели:
- Использование T5ForConditionalGeneration, мощной модели для условной генерации текста, предварительно обученной на различных задачах.

2. Предобработка данных:
- Разбиение текста на батчи по его длине, токенизация и последующая подготовка для обучения модели.

3. Дообучение модели:
- Процесс дообучения модели на выбранном корпусе текста, что позволяет адаптировать ее к особенностям русскоязычных новостных статей.

4. Оптимизация времени выполнения:
- Для ускорения процесса, использование заранее дообученной модели, позволяя сократить время ожидания.

### **Процесс работы:**

1. Ввод текста:
  - Пользователь вводит текст на русском языке, требующий сжатия.

2. Обработка текста:
 - Текст разбивается на батчи (по умолчанию установлен кооэфициент раывны 64 для реализации кода на 15Gb видеопамяти Colab), токенизируется и подготавливается для предсказания на основе обученной модели.

3. Сжатие текста:
 - Модель T5ForConditionalGeneration генерирует 1-3 содержательных предложения, отражающих главную суть введенного текста.

4.  Вывод результата:
 - Сжатый текст выводится в виде краткого содержания или резюме.

### **Преимущества:**

- [x] Автоматизация процесса сжатия текста.
- [x] Эффективное извлечение ключевой информации из больших объемов статей.
- [x] Использование заранее дообученной модели для оптимизации времени выполнения.

### **Важная информация:**
# Из-за ограничений файлов GitHub, а так же для экономии времени дообученая модель скачивается с HuggingFace! В проекте выделен код, который отвечал за дообучение модели.
# Размер файлом модели До
![До](https://github.com/DyKiVy/Lab_text_summarization/blob/main/%D0%92%D0%B5%D1%81%20%D0%B4%D0%BE.png)
# Размер файлом модели После
![После](https://github.com/DyKiVy/Lab_text_summarization/blob/main/%D0%92%D0%B5%D1%81%20%D0%BF%D0%BE%D1%81%D0%BB%D0%B5.png)
### **Ссылка на файл в [Colab](https://colab.research.google.com/drive/1-YAL2h6rbM7vLwuOvVS24jZMRhbfkTnP#scrollTo=evOtSwsb3Lsu)**

### **Примеры :**
#### Текст 1.
Переселение россиян из аварийного жилищного фонда в новые дома — важная часть программы национального проекта «Жилье и городская среда», созданного по решению президента. В 2023 году благодаря этой программе в новые квартиры переехало более 100 тыс. человек, а всего расселили 1,85 млн кв. м непригодного для проживания жилья. О том, в каких регионах преуспела программа и как о ней отзываются местные жители — в материале «Газеты.Ru». С 2019 года в России реализуется федеральный проект «Обеспечение устойчивого сокращения непригодного для проживания жилищного фонда», входящего в нацпроект «Жилье и городская среда». По нему за это время удалось расселить 9,8 млн кв. м аварийного жилья — из аварийных домов в новые квартиры переехало более 578 тыс. человек. При этом за 10 месяцев текущего года по программе переезд в новые дома осуществили 106,6 тыс. человек — всего в этом году получилось расселить 1,85 млн кв. м аварийного жилья. Российские регионы активно расселяют граждан из ветхого жилья, и некоторые субъекты делают это с опережением графика, несмотря на удорожание строительства из-за роста цен на земельные участки, строительные материалы и высокие затраты на благоустройство, что связано с объективными причинами на рынке недвижимости. Нацпроектом было запланировано до конца 2024 года расселить 536 560 человек — это значит, что текущие результаты уже опередили намеченные цели. «Такое перевыполнение стало возможным в том числе благодаря опережающему финансированию, одобренному правительством. То есть для регионов появилась возможность оперативно получить средства, предусмотренные только на следующие годы, что позволило быстрее переселять людей. На сегодня действующую программу, которая предусматривает расселение аварийного жилья, выявленного до 2017 года, завершили 16 субъектов Российской Федерации, и до конца года к ним присоединится еще 34 региона. Остальные субъекты продолжают выполнение программы», — рассказал заместитель председателя правительства Марат Хуснуллин. Как рассказал министр строительства и жилищно-коммунального хозяйства Ирек Файзуллин, чтобы важная социально-значимая работа по переселению людей не останавливалась, в 2022 году запустили новую программу, к которой приступили 14 субъектов — по ней из жилья общей площадью 259 тысяч кв. м уже переехали почти 13 тыс. человек. Кроме того, в сентябре 2023 года по поручению Владимира Путина приняли постановление о создании с 1 января 2024 года условий для ускоренного расселения людей из ветхого жилья.ь Если раньше регионам сначала нужно было завершить первый этап и расселить жилье, которое было признано аварийным до 2017 года, и только потом перейти к следующему этапу, то теперь реализовывать обе программы можно параллельно, если ликвидация непригодного жилья осуществляется в рамках проектов комплексного развития территорий (КРТ). Финансируется программа нацпроекта по переселению из аварийного жилого фонда за счет средств федерального бюджета, консолидированных бюджетов Российской Федерации, а также из внебюджетных источников. При этом некоторые субъекты дополнительно расселяют аварийное жилье за счет средств своих бюджетов. Так, из отмеченных выше полумиллиона человек более 150 тыс. переехали в новые квартиры по собственным программам субъектов. Регионом-лидером по переселению граждан из непригодного жилья стал Ханты-Мансийский автономный округ, где за время реализации программы расселено более 44 тыс. человек. На втором месте — Пермский край, где из ветхих домов переехали более 40 тыс. жителей. Закрыл тройку лидеров Ямало-Ненецкий автономный округ, где аварийное жилье на новое сменили порядка 38 тыс. человек. При этом в Ханты-Мансийском округе расселение аварийного жилищного фонда продолжает идти опережающими темпами. За 2023 год здесь переехали в новые дома 8472 человека из более чем 130 тыс. кв. м ветхого жилья. В 2024 году планируется расселить еще 63 тыс. кв. м аварийного фонда. В начале этого года по программе нацпроекта «Жилье и городская среда» переехала семья Марии Сергеевны Колодько из Сургута. Она жила в квартире, которая находилась в двухэтажном деревянном доме, который признали аварийным еще в 1989 году.В нем не было капитального ремонта, но были деревянные полы, которые прогнили и рассохлись от погодных условий, дом осел, а в окнах появились огромные щели. Также в квартире были постоянные проблемы с горячей водой, и обогревательные котлы приходилось постоянно менять из-за плохих труб. «Почти 34 года мы жили надеждой на переселение. И только 5 лет назад, когда запустилась программа по расселению из аварийного жилья, начали вырисовываться перспективы на переезд. Два года назад мы начали собирать необходимые документы и уже в феврале этого года переехали в новую квартиру в многоэтажном доме. Новая квартира площадью 51 кв.м — на 15 метров больше нашей старой. Многоэтажный дом построен по современным технологиям, в квартире установлен двухфазный счетчик, бойлер на 100 литров воды. Проблем с водой у нас за 8 месяцев ни разу не было, воды хватает. А семья у нас немаленькая: я с мужем и дочкой и мама с отчимом. Мы все очень довольны», — рассказала свою историю Мария Колодько. До конца 2023 года досрочно завершить программу по переселению граждан планируют и в Солнечногорске. В комфортные квартиры вот-вот переедут более 900 человек. Среди них — семья Натальи и Евгения Рябининых, которые вынуждены снимать квартиру, поскольку в их собственном жилье прогнили полы, обветшала кровля и давно нет горячей воды. «Большое спасибо администрации муниципалитета за то, что приложили все усилия для улучшения наших жилищных условий. Реальность оказалась лучше всех ожиданий. Здесь просторные комнаты, в том числе и кухня, большой коридор, два санузла и застекленный балкон. Все увиденное нас с мужем очень порадовало», — поделилась своими эмоциями Наталья Рябинина. Супругам уже показали квартиру в новостройке, которую вскоре им передадут по программе переселения из ветхого жилья.«Расселение аварийных домов направлено в первую очередь на улучшение жилищных условий граждан и повышение комфорта и безопасности их проживания. Мероприятия программы в целом выполняются хорошими темпами», — добавил генеральный директор «Фонда развития территорий» Ильшат Шагиахметов. Напомним, что программа переселения граждан из аварийного жилищного фонда реализуется по инициативе президента Владимира Путина с 2008 года. Оператором программы выступает ППК «Фонд развития территорий». За этот период жилищные условия улучшили 1,64 млн россиян. С 2019 года переселение из ветхих домов осуществляется в рамках национального проекта «Жилье и городская среда», который также реализуется по поручению президента. Главная цель нацпроекта — улучшить жилищные условия и создать комфортную и безопасную среду для жизни.

#### Ответ.
Программа национального проекта «Жилье и городская среда» по расселению россиян из аварийного жилищного фонда в новые дома преуспела. В этом году по программе удалось расселить 1,85 млн кв. м непригодного для проживания жилья. Это значит, что текущие результаты уже опередили планы, несмотря на удорожание строительства.

#### Текст 2.
Если бы Достоевский жил в наше время, смотрел по вечерам «Черное зеркало» и просто читал новости, то, скорее всего, Раскольникова судил бы Искусственный интеллект. Сейчас довольно сложно представить, как будет выглядеть судебный процесс будущего, где условному судье не придется протирать 6 лет штаны на институтской скамье и насильно впихивать в свой мозг тонны законодательной базы. Кажется, это будет что-то коллективное и похожее на продвинутых присяжных с «закаченными» правовыми нормами, чтобы моральные аспекты при этом оставались на более близкой человеческой стороне.Привет, Хабр, это команда  Alliesverse – платформы для управления бизнесом – и это в наши воспаленные мозги попала идея о современном Раскольникове... Случилось это на большом мероприятии, посвященному ChatGPT, на которое нас пригласили. Мы подумали: а что если ускорить наступление высокотехнологичного и справедливого суда и попробовать обучить ИИ всем российским кодексам ?Так сформировался эмбрион LawAi by Alliesverse. Под катом, подобно ChatGPT, расскажем наш опыт обучения русскоязычных и зарубежных моделей нейросетей российскому законодательству.

#### Ответ.
Если бы Достоевский жил в наше время, то Раскольникова судил бы Искусственный интеллект, а если бы он жил на институтской скамье и просто читал новости, то это было бы справедливое судебное разбирательство.

#### Текст 3.
"Дмитрий Иванович был учителем в техникуме. В тот день он пришел домой, аккуратно расставил все продукты в холодильник, которые принес в сумке. Расстелил скатерть на столе. Достал фужер из стенного шкафа. Рядом положил блюдце с нарезанным сыром. Бутылку вина Мерло, крепкое полусладкое. ВЫключил свет. Включил телевизор. Налил полный фужер. Он приготовился провести очередной приятный вечер в полном одиночестве. Через неделю после того как случилась катастрофа он уже легко резал людей ножом, которым когда то резал сыр. Он уже легко пил водку, и забыл название любимой телепередачи. А команда Зенит, за которую он когда то болел - для него стала чем то мифическим, из другого мира. И тех людей, которым он ставил когда то оценки в классе, он убивал не задумываясь. И даже с каким то удовольствием и с некоторой долей самоотдачи и смекалки...

#### Ответ.
Дмитрий Иванович, который когда то резал людей ножом и пил водку, забыл название своей любимой телепередачи, а команда Зенит - для него стала чем то мифическим.

#### Текст 4.
Когда человек рождается он начинает проходить процесс развития, познавать открывшийся ему мир, ему раскрываются смысл и ценности жизни. Иными словами, жизненный путь человека начинается с развития его личности. Данный процесс многогранен и для каждого из нас является индивидуальным. Каждый человек обладает своим собственным мировоззрением и, далеко не каждый, включает в него занятия физическими нагрузками, что является важной частью здоровья. Первой и самой важной потребностью для человека является его здоровью, поскольку благодаря нему, человек правильно проходит процесс развития, а от его состояния зависит трудоспособность человека. Для того, чтобы здоровье было на более высоком уровне, человеку необходимо придерживаться здоровому образу жизни, который включает в себя основные такие элементы, как: правильное питание, гигиену, труд и отдых, а также самый важный элемент — правильную двигательную активность.

#### Ответ.
Когда человек рождается, он начинает проходить процесс развития, познавать открывшийся ему мир, ему раскрываются смысл и ценности жизни, а от его состояния зависит трудоспособность человека. Для того, чтобы здоровье было на более высоком уровне, необходимо придерживаться здоровому образу жизни, который включает в себя основные элементы здоровья.
