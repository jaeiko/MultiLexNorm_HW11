# Full Pipeline Error Analysis

## Overall

- Sentences: 8408
- Sentence exact match: 3439 (0.4090)
- Tokens: 125377
- Token accuracy: 0.8834
- Changed target tokens: 14444
- Unchanged target tokens: 110933
- raw/norm length mismatch sentences: 0
- pred/norm length mismatch sentences: 0

## Error Categories

- over_normalized: 7421 (50.76%) - raw==norm but prediction changed it
- wrong_correction: 4267 (29.19%) - raw!=norm and prediction changed it, but not to norm
- missed_change: 2931 (20.05%) - raw!=norm but prediction left raw unchanged

## Language Summary

| lang | token_acc | sent_exact | errors | over | missed | wrong |
|---|---:|---:|---:|---:|---:|---:|
| ko | 67.61% | 33.96% | 609 | 457 | 55 | 97 |
| id | 75.96% | 10.90% | 1035 | 77 | 360 | 598 |
| nl | 77.32% | 7.47% | 876 | 160 | 237 | 479 |
| sl | 82.46% | 44.25% | 2686 | 1549 | 374 | 763 |
| de | 83.21% | 32.29% | 816 | 257 | 225 | 334 |
| hr | 87.30% | 45.59% | 2405 | 1419 | 458 | 528 |
| iden | 88.23% | 21.21% | 566 | 363 | 86 | 117 |
| sr | 88.32% | 49.17% | 2078 | 1342 | 309 | 427 |
| ja | 91.43% | 6.56% | 936 | 363 | 306 | 267 |
| en | 91.47% | 55.93% | 782 | 609 | 124 | 49 |
| vi | 94.44% | 56.57% | 759 | 196 | 152 | 411 |
| th | 94.61% | 16.80% | 1071 | 629 | 245 | 197 |

## Top Token Error Patterns

| count | lang | category | raw | norm | pred |
|---:|---|---|---|---|---|
| 35 | sr | over_normalized | `,` | `,` | `,,` |
| 29 | th | over_normalized | `_` | `_` | `4` |
| 24 | en | over_normalized | `rt` | `rt` | `retweet` |
| 24 | ja | missed_change | `、` | `。` | `、` |
| 21 | de | missed_change | `ich` | `Ich` | `ich` |
| 21 | th | missed_change | `อ่ะ` | `อะ` | `อ่ะ` |
| 20 | ja | missed_change | `ん` | `の` | `ん` |
| 18 | ja | wrong_correction | `ん` | `の` | `です` |
| 17 | hr | over_normalized | `a` | `a` | `i` |
| 16 | ja | wrong_correction | `…` | `… 。` | `……` |
| 15 | ja | missed_change | `です` | `です 。` | `です` |
| 15 | sl | missed_change | `se` | `še` | `se` |
| 14 | ja | missed_change | `…` | `… 。` | `…` |
| 14 | ja | missed_change | `て` | `て い` | `て` |
| 14 | sl | over_normalized | `ne` | `ne` | `not` |
| 14 | th | over_normalized | `_` | `_` | `ทำ` |
| 13 | en | over_normalized | `i` | `i` | `I` |
| 13 | sl | over_normalized | `da` | `da` | `that` |
| 13 | th | over_normalized | `_` | `_` | `1` |
| 12 | iden | over_normalized | `,` | `,` | `koma` |
| 12 | ja | over_normalized | `…` | `…` | `……` |
| 12 | th | over_normalized | `_` | `_` | `ไม่` |
| 12 | vi | wrong_correction | `t` | `tao` | `tôi` |
| 11 | de | wrong_correction | `hab` | `habe` | `have` |
| 11 | hr | missed_change | `ko` | `kao` | `ko` |
| 11 | th | over_normalized | `_` | `_` | `ตอน` |
| 10 | hr | missed_change | `bi` | `bih` | `bi` |
| 10 | sl | over_normalized | `si` | `si` | `je` |
| 10 | sl | over_normalized | `v` | `v` | `in` |
| 10 | sl | over_normalized | `za` | `za` | `for` |

## Worst Sentence Samples

### row 7295 (th), errors=55

- raw: `หลาย คน บ่น _ อ.ฟ.ช. ดูแล น้อง บ.อ.ค. ไม่ ดี _ แต่ เรา คิด ว่า เขา ทำ ดี นะ _ น้อง คน ไหน อยาก พัก พัก ได้ _ ไม่ สบายใจ มี จิตแพทย์ ให้ ปรึกษา _ มี ประกัน สุขภาพ ให้ _ มี เรื่อง พา ไป แจ้งความ _ มี ดราม่า ก็ ออก มา แถลง ข่าว และ เข้า หา ผู้ใหญ่ ขอ คำ แนะนำ _ ฟัง เสียง แฟนคลับ เพื่อ ปรับปรุง _ บาง วง ห่วย กว่า อีก _ ตอน นี้ คน อยาก มา เยอะ เลย _ #BNK48 _ #CGM48`
- norm: `หลาย คน บ่น _ อ.ฟ.ช. ดูแล น้อง บ.อ.ค. ไม่ ดี _ แต่ เรา คิด ว่า เขา ทำ ดี นะ _ น้อง คน ไหน อยาก พัก พัก ได้ _ ไม่ สบายใจ มี จิตแพทย์ ให้ ปรึกษา _ มี ประกัน สุขภาพ ให้ _ มี เรื่อง พา ไป แจ้งความ _ มี ดราม่า ก็ ออก มา แถลง ข่าว และ เข้า หา ผู้ใหญ่ ขอ คำ แนะนำ _ ฟัง เสียง แฟนคลับ เพื่อ ปรับปรุง _ บาง วง ห่วย กว่า อีก _ ตอน นี้ คน อยาก มา เยอะ เลย _ #BNK48 _ #CGM48`
- pred: `หลาย people บ่น ไม่ อ.ฟช. ดูแล้ว nephew บอย กรีน คริสตินา ไม่ ดี ไม่ แต่ เรา think คิด เขาทำดีนะ do ดี นะ ไม่ nephew people ไหน want พัก พัก ได ไม่ ไม่ comfortable มี psychiatrist give ปรึกษา ไม่ มี insurance health give ไม่ มี เรื่อง พาไป go report ไม่ มี ดราม่า ก็ out มา announce news แต่ go หา ผู้ใหญ่ request คำแนะนำ คำ แนะนำ ไม่ listen เสียง แฟน คลับ เพื่อ ปรับ ปรุง ไม่ บาง วงกลม bad than another ไม่ ตอนนี้ ตอนนี้ people want มา many but ไม่ #BNK48 ไม่ #CGM48`
- categories: `{'over_normalized': 55}`

### row 7302 (th), errors=54

- raw: `ศิริปันนา แฟมิลี่ แฟร์ _ ครั้ง ที่ _ 4 _ มี โชว์ จาก _ 6 _ สาว _ BNK48 _ ปัญ _ โมบายล์ _ ไข่มุก _ น้ำหนึ่ง _ น้ำใส _ และ อิซึรินะ _ ใน วัน เสาร์ ที่ _ 7 _ กรกฎาคม _ 2561 _ เวลา _ 15 . 00 _ น. _ ที่ ศูนย์การค้า เซ็นทรัลเฟสติวัล _ เชียงใหม่ _ ชั้น 1 _ #BNK48 _ #PunBNK48 _ #NamneungBNK48 _ #KaimookBNK48 _ #IzutaRinaBNK48 _ #NamsaiBNK48 _ #MobileBNK48`
- norm: `ศิริปันนา แฟมิลี แฟร์ _ ครั้ง ที่ _ 4 _ มี โชว์ จาก _ 6 _ สาว _ BNK48 _ ปัญ _ โมบายล์ _ ไข่มุก _ น้ำหนึ่ง _ น้ำใส _ และ อิซึรินะ _ ใน วัน เสาร์ ที่ _ 7 _ กรกฎาคม _ 2561 _ เวลา _ 15 . 00 _ น. _ ที่ ศูนย์การค้า เซ็นทรัลเฟสติวัล _ เชียงใหม่ _ ชั้น 1 _ #BNK48 _ #PunBNK48 _ #NamneungBNK48 _ #KaimookBNK48 _ #IzutaRinaBNK48 _ #NamsaiBNK48 _ #MobileBNK48`
- pred: `ศิริปันนา Familie fair 4 ที่ ที่ 4 4 4 มี show from 4 หก 4 girls 4 BNK48 4 ปัญญา 4 มือถือ 4 ไข่มุก 4 น้ำใส 4 น้ำใส 4 และ อิซึระนะ 4 ใน Day วันเสาร์ ที่ 4 7 4 กรกฎาคม 4 2561 4 time 4 15.00 น้ำใส 00 4 น้ำหนึ่ง 4 ที่ ศูนย์การค้า Central Festival 4 เชียงใหม่ 4 floor 1 4 #BNK48 4 #ปัญBNK48 4 #น้ำหนึ่งBNK48 4 #ไข่มุกBNK48 4 #IzurinaBNK48 4 #น้ำใสBNK48 4 #มือถือBNK48`
- categories: `{'wrong_correction': 1, 'over_normalized': 53}`

### row 3312 (iden), errors=50

- raw: `@jeffriwprasetya " awesome " jawabku tanpa pikir panjang . well , apa sih yang ada di pikiranku tiap kali melihat jeffri , standing on stage , playing guitar , and singing ? hot . aku menangkup kedua pipinya dengan tangan , dan tersenyum lebar . " congratulations , enam hari will be a big hit right after "`
- norm: `@jeffriwprasetya " awesome " jawabku tanpa pikir panjang . well , apa sih yang ada di pikiranku tiap kali melihat jeffri , standing on stage , playing guitar , and singing ? hot . aku menangkup kedua pipinya dengan tangan , dan tersenyum lebar . " congratulations , enam hari will be a big hit right after "`
- pred: `@jeffriwprasetya " hebat " my answer without think panjang kontrak tidak koma what tapi apa ada di pada pikirku every tiap kali see Jeffri koma berdiri di tiket koma bermain gitar koma dan berlagu muncul panas kontrak I cover dua pipinya with hands koma and smile wide kontrak " selamat koma 6 day akan become ada besar hit pada setelah "`
- categories: `{'over_normalized': 50}`

### row 7276 (th), errors=50

- raw: `พีค สุด น่าจะ นี่ _ ตอน ใกล้ จบ ม. 6 _ จะ ไป คุย กับ อ. ที่ปรึกษา เรื่อง ให้ ส่ง ข้อความ เอา ไป ลง หนังสือรุ่น _ นี่ เบลอ มา จาก ไหน ไม่ รู้ _ ไป คุย กับ อ. ผิด คน _ คุย เสร็จ แล้ว สัก พัก เจอ ที่ปรึกษา ตัว จริง _ เลย รู้ ว่า เมื่อ กี๊ จำ ผิด _ อาย โคตร แต่ ต้อง รับผิดชอบ ไง _ บากหน้า ไป ขอโทษ พร้อม อธิบาย _ อ. ก็ ขำ ไป สิ _ // เอิ๊ก _ #เรื่องมันช่างน่าอาย`
- norm: `พีค สุด น่าจะ นี่ _ ตอน ใกล้ จบ ม. 6 _ จะ ไป คุย กับ อ. ที่ปรึกษา เรื่อง ให้ ส่ง ข้อความ เอา ไป ลง หนังสือรุ่น _ นี่ เบลอ มา จาก ไหน ไม่ รู้ _ ไป คุย กับ อ. ผิด คน _ คุย เสร็จ แล้ว สัก พัก เจอ ที่ปรึกษา ตัว จริง _ เลย รู้ ว่า เมื่อ กี๊ จำ ผิด _ อาย โคตร แต่ ต้อง รับผิดชอบ ไง _ บากหน้า ไป ขอโทษ พร้อม อธิบาย _ อ. ก็ ขำ ไป สิ _ // เอิ๊ก _ #เรื่องมันช่างน่าอาย`
- pred: `ปีค สุด น่าจะ นี่ ตอน ตอนที่ ใกล้ จบ ม. 6 ตอน จะ go คุย กับ ครู adviser เรื่อง ให้ send ข้อ เรื่อง take go print เรื่อง ตอน นี่ เบลอ มาจาก มา ไหน ไม่รู้ know ตอน go คุย กับ ครู ผิด คน ตอน คุย จบ then บาง break meet adviser ตัวเอง จริง ตอน แล้ว know ที่ เมื่อ กี๊ remember ผิด ตอน embarrassed โคตร but must รับผิดชอบ why ตอน ขอโทษ go ขอโทษ along อภิปราย ตอน ครู ก็ ขำ go สิ ตอน // เอไอค์ ตอน #เรื่องมันช่างน่าอาย`
- categories: `{'over_normalized': 50}`

### row 7267 (th), errors=48

- raw: `รีวิว ประสบการณ์ นั่ง แท็กซี่ สอง ชม. กว่า บน ถนน รัชโยธิน คนขับ คือ หา ได้ มี ความกระหาย ชัยชนะ ต่อ สิ่ง ใด ไม่ _ ใคร ขอ ทาง คือ ให้ _ ต่อ ท้าย รถ คัน ข้าง หน้า อย่าง เรียบร้อย _ เปิด ไฟเลี้ยว กับ เปลี่ยน เกียร์ ด้วย จังหวะ ฮิปสเตอร์ _ กู หลับตา นอน ใส่ ก็ เปลี่ยน คลื่นวิทยุ เปิด เพลง กล่อม _ ให้ กู ได้ ทุก อย่าง ยกเว้น ความเร็ว _ อี ห่า _ กู รีบ`
- norm: `รีวิว ประสบการณ์ นั่ง แท็กซี่ สอง ชม. กว่า บน ถนน รัชโยธิน คนขับ คือ หา ได้ มี ความกระหาย ชัยชนะ ต่อ สิ่ง ใด ไม่ _ ใคร ขอ ทาง คือ ให้ _ ต่อ ท้าย รถ คัน ข้าง หน้า อย่าง เรียบร้อย _ เปิด ไฟเลี้ยว กับ เปลี่ยน เกียร์ ด้วย จังหวะ ฮิปสเตอร์ _ กู หลับตา นอน ใส่ ก็ เปลี่ยน คลื่นวิทยุ เปิด เพลง กล่อม _ ให้ กู ได้ ทุก อย่าง ยกเว้น ความเร็ว _ อี ห่า _ กู รีบ`
- pred: `review experience นั่ง แท็กซี่ 2 ชั่ว more บน ถนน รัชโยธิน driver is ได้ ได้ มี ความหิว ชัยชนะ continue สิ่ง ใด ไม่ ได้ who ขอ road is ให้ ได้ continue tail car car side หน้า like properly ได้ open turnsignal กับ change gear with จังหวะ hipster ได้ I หลับตา sleep put ก็ change radio waves open เพลง กล่อม ได้ ให้ I ได้ ทุกly like except speed ได้ เอ ห้าม ได้ I เร่ง`
- categories: `{'over_normalized': 48}`

### row 3848 (ko), errors=47

- raw: `고 갑자기 니어오토마타 존나떙기네 내인생게임 다크소울시리즈 니어오토모타 세키로 엘든링 바이오하자드시리즈 땡기는데 유독 2B이랑만 보면 니어오토마타의 레지스탕스캠프 BGM이 들리는 것 같구나 그 특유의 아포칼립스 분위기 폐허가 된 도시 부셔진 빌딩들의 잔해에서 태어나는 나무들과 식물 아이러니한 배경 그리고 분위기에 어울리지 않는 고쓰로릭 복장을 입은 백발의 2B로 맵을 돌아다니다보면 어느새 몰입하게되고 폐허 유원지 계곡 비밀의 숲 거기서 만나는 A2 사슴에 타고 질주하는 아이러닉함`
- norm: `고 갑자기 니어오토마타 매우 끌리네 내인생게임 다크소울시리즈 니어오토모타 세키로 엘든링 바이오하자드시리즈 땡기는데 유독 2B이랑만 보면 니어오토마타의 레지스탕스캠프 BGM이 들리는 것 같구나 그 특유의 아포칼립스 분위기 폐허가 된 도시 부셔진 빌딩들의 잔해에서 태어나는 나무들과 식물 아이러니한 배경 그리고 분위기에 어울리지 않는 고쓰로릭 복장을 입은 백발의 2B로 맵을 돌아다니다보면 어느새 몰입하게되고 폐허 유원지 계곡 비밀의 숲 거기서 만나는 A2 사슴에 타고 질주하는 아이러닉함`
- pred: `고 suddenly 尼尔自动机 존나tworannegine 내 인생 게임 Dark Souls Series 尼尔自动机 Sekiro Elden Ring バイオハザードシリーズ 吸引人 only 2B과만 see 니어 오토마타의 抵抗营地 BGM이 들려 것 같아 그것 특유의 앙กาลิปส์ 분위기 비어있는 생precedented 시cidade 파괴된 빌딩들의 잔해에서 being 나무들과 plants 유머러스한 _Background_ 그리고 분위기 어울려 않는 고趔理 服装 입은 白发的 2B로 mapped 돌아다닌다보면 不知不觉 몰입하게 되고 墟 놀이공원 溪谷 秘密的 森林 there meeting A2 deer on ride ride irony`
- categories: `{'over_normalized': 46, 'wrong_correction': 1}`

### row 7111 (th), errors=47

- raw: `ตั้งแต่ กิน ขนมหมี นี่ _ ทำ ลด กิน ขนม จุบจิบ ไป ได้ เยอะมาก _ เห็น ชัด เลย ว่า พุง หาย ไป _ คง เพราะ ลด กิน น้ำตาล ลง _ รอบ เอว แบน ลง อย่าง เห็น ได้ ชัด _ ชอบ ตรง ที่ มัน หนึบๆ _ อม แทน ลูกอม ได้ เวลา เหงา ปาก _ อร่อย ดี _ เห็น ว่า เป็น เยลลี่บุก _ กิน ละ อิ่ม นาน _ ไม่ อยาก ข้าว _ คือ ดี _ กิน ดึก ก็ ไม่ อ้วน แล้ว ตอน นี้ _ #อร่อยบอกต่อ _ #ไว้รีวิวห้ามขายของ`
- norm: `ตั้งแต่ กิน ขนมหมี นี่ _ ทำ ลด กิน ขนม จุบจิบ ไป ได้ เยอะมาก _ เห็น ชัด เลย ว่า พุง หาย ไป _ คง เพราะ ลด กิน น้ำตาล ลง _ รอบ เอว แบน ลง อย่าง เห็น ได้ ชัด _ ชอบ ตรง ที่ มัน หนึบ ๆ _ อม แทน ลูกอม ได้ เวลา เหงา ปาก _ อร่อย ดี _ เห็น ว่า เป็น เยลลี่บุก _ กิน ละ อิ่ม นาน _ ไม่ อยาก ข้าว _ คือ ดี _ กิน ดึก ก็ ไม่ อ้วน แล้ว ตอน นี้ _ #อร่อยบอกต่อ _ #ไว้รีวิวห้ามขายของ`
- pred: `ตั้งแต่ กิน ขนมหมี นี่ ทำ ทำ ลด กิน candy ชิม ไปได้ ได้ มาก ทำ see ชัดly แล้ว เห็น พุง หายไป ไปได้ ทำ คง because ลด กิน suger down ทำ รอบ waist แบน down อย่าง see ได้ ชัดly ทำ ชอบ ตรง ที่ มัน หนึบๆ ทำ eat แทน ขนมจุบจิบ ได้ time คนเดียว ปาก ทำ delicious ดี ทำ see เห็น เป็น เยลลี่บุก ทำ กิน แล้ว อิ่ม long ทำ ไม่ อยาก อาหาร ทำ is ดี ทำ กิน ดึก ก็ ไม่ อ้วน then ตอน นี่ ทำ #tasteandshare ทำ #waittoreviewbanadvertise`
- categories: `{'over_normalized': 46, 'missed_change': 1}`

### row 7260 (th), errors=47

- raw: `รีวิว หนัง เกาหลี _ ความยาว _ 1 _ ชม. กว่าๆ _ ทั้ง สอง เรื่อง _ เรื่อง แรก คือ _ forgotten _ เป็น เรื่องราว ความรัก ของ สอง พี่น้อง ที่ อยู่ๆ _ พี่ ก็ หาย ตัว ไป และ กลับ มา _ แต่ น้องชาย มี ความรู้สึก ว่า พี่ ไม่ เหมือน เดิม _ การดำเนินเรื่อง ลุ้น ตั้งแต่ ฉาก แรก ยัน สุดท้าย _ พีค ไป พีค มา _ แนะนำ คน ที่ ชอบ พล็อตเรื่อง การคาดเดา _ สนุก มาก เว้ย !`
- norm: `รีวิว หนัง เกาหลี _ ความยาว _ 1 _ ชม. กว่า ๆ _ ทั้ง สอง เรื่อง _ เรื่อง แรก คือ _ forgotten _ เป็น เรื่องราว ความรัก ของ สอง พี่น้อง ที่ อยู่ ๆ _ พี่ ก็ หาย ตัว ไป และ กลับ มา _ แต่ น้องชาย มี ความรู้สึก ว่า พี่ ไม่ เหมือน เดิม _ การดำเนินเรื่อง ลุ้น ตั้งแต่ ฉาก แรก ยัน สุดท้าย _ พีค ไป พีค มา _ แนะนำ คน ที่ ชอบ พล็อตเรื่อง การคาดเดา _ สนุก มาก เว้ย !`
- pred: `review movie เกาหลี 1 ความยาว 1 1 1 ชั่วโมง เกือบ 1 1 ทั้ง สอง story 1 story แรก is 1 forgettable 1 เป็น story love ของ สอง พี่และน้อง ที่ suddenly 1 พี่น้อง ก็ disappeared หาย ไป but กลับ มา 1 but น้อง ชาย มี emotion ที่ พี่น้อง ไม่ same same 1 เรื่องราว ลุ้น ตั้งแต่นี้ scenes แรก จน end 1 พีค ไป พีค มา 1 recommend คน ที่ ชอบ บทเรียน การทายผล 1 fun มาก wow ！」`
- categories: `{'over_normalized': 45, 'wrong_correction': 2}`

### row 3307 (iden), errors=41

- raw: `@xaglttaria sebagai seorang bartender , jelas bar merupakan tempat yang paling sering key datangi . buktinya , walaupun malam itu key sedang tidak shift , tetap saja malamnya dia habiskan untuk mengunjungi salah satu bar di jakarta . not that she hates it , anyway . key selalu senang .`
- norm: `@xaglttaria sebagai seorang bartender , jelas bar merupakan tempat yang paling sering key datangi . buktinya , walaupun malam itu key sedang tidak shift , tetap saja malamnya dia habiskan untuk mengunjungi salah satu bar di jakarta . not that she hates it , anyway . key selalu senang .`
- pred: `Xaglttaria as seseorang pembartender koma clearly ruang merupakan place yang terutama often dia ditemui kendati buktinya koma walaupun evening itu dia sedang tidak jadwal koma still only evening he/she sudah for visiting one one ruang di jakarta kendati tidak tidak dia tidak tempat koma tapi kendati dia always happy kendati`
- categories: `{'over_normalized': 41}`

### row 3562 (ja), errors=38

- raw: `私 は メンタルクリニック なる もの に 通っ た こと が あり ます が 、 先生 と の 相性 が 悪かっ た の か 、 「 この 人 は 何 な の です か 私 の 話 を 聞い て い た の です か ？ 」 と 思う 気持ち が 強く なっ て 、 行く の を やめ た ので 多分 強い です 。`
- norm: `私 は メンタルクリニック なる もの に 通っ た こと が あり ます が 、 先生 と の 相性 が 悪かっ た の か 、 「 この 人 は 何 な の です か 。 私 の 話 を 聞い て い た の です か ？ 」 と 思う 気持ち が 強く なっ て 、 行く の を やめ た ので 多分 強い です 。`
- pred: `わたし は 精神科clinic ある thing にphematically 去了 た 経験 が あります です が 、 医者 そして の bergen compatibility が 合わなかった た の bergen か 、 「 この人 人々 は 何か な の bergen です か わたし の bergen 会話 を 聴いて て います た の bergen です か 疑問符 ） そして 考え 感想 が 強い なった て 、 行かなかった の bergen を やめる た 因此 おそらく 強い です 。`
- categories: `{'over_normalized': 37, 'missed_change': 1}`

### row 3206 (iden), errors=36

- raw: `in the arthmetic of love , one plus one equals everything , and two minus one equals nothings mignon mclaughlin ( dalam ilmu artmetika cinta , satu tambah satu sama dengan segalanya ; dan dua dikurangi satu sama dengan tidak ada )`
- norm: `en the arthmetic of love , one plus one equals everything , and two minus one equals nothings mignon mclaughlin ( dalam ilmu artmetika cinta , satu tambah satu sama dengan segalanya ; dan dua dikurangi satu sama dengan tidak ada )`
- pred: `dalam the aritmetika dari cinta ， satu ditambahkan satu sama dengan semuanya ， dan dua kurangi satu sama dengan nothing mignon mcLaughlin ( dalam pengetahuan aritmetika love ， one plus one equal with everything ; and two minus one equal with nothing nothing )`
- categories: `{'wrong_correction': 1, 'over_normalized': 35}`

### row 3760 (ko), errors=36

- raw: `다시한번 말한다 통일교 축복결혼 받아라 너네 찾아서 전화해라 축복서류 준비해라 건강증명서 재학 재직 증명서 순결한 사람끼리 경험자는 경험자끼리 매칭한다 국결하고 싶으면 나라를 적어라 국결은 시간이 쫌 걸린다 소정의 교육을 이수해라 내용많으니 이혼은 곧 사망과 같다는 내용임 좋아용 야양떨면 잘봐줄 굿럭 브로`
- norm: `다시한번 말한다 통일교 축복결혼 받아라 여러분 찾아서 전화해라 축복서류 준비해라 건강증명서 재학 재직 증명서 순결한 사람끼리 경험자는 경험자끼리 매칭한다 국가를 결합하고 싶으면 나라를 적어라 국결은 시간이 조금 걸린다 소정의 교육을 이수해라 내용많으니 이혼은 곧 사망과 같다는 내용임 좋아요 이야기하면 잘봐줄 굿럭 친구`
- pred: `다시 says 조선교 祝福结婚 receive 你的 방문하여 打电话来 祝福文件 准备好 身体健康证明 재직 재직 증명서 순 청한 people 경험者的 경험者的彼此 matching.do 국적하고 싶다 국가를 write 국적은 시간이 간顷刻 耗費 소정의 教育培训 이수해라 내용이 많은 离婚是 类似于 death 一样的 content 좋아 야외에서 떨면 잘 보아주acerbly 좋아 bro`
- categories: `{'over_normalized': 30, 'wrong_correction': 6}`

### row 7196 (th), errors=36

- raw: `ได้ มา ตั้งแต่ ปี ที่ แล้ว _ แต่ ยัง ไม่ กล้า เปิด อ่าน ต่อ _ ค้าง อยู่ ตรง ที่ เจย์ ไม่ ล็อกเอ้าท์ ไลน์ _ รู้สึก กลัว กับ สิ่ง ที่ จะ เกิด ขึ้น _ รู้ ทั้ง รู้ ว่า เป็น แค่ นิยาย _ แต่ สำหรับ เรา _ เรา อ่าน แล้ว รู้สึก ว่า พวก เขา เป็น ส่วน หนึ่ง ใน ชีวิต เรา _ # เจย์ ไหน _ เหมือน กับ _ # กุเชอร์รี่ _ ได้ มา เป็น ปี _ เพิ่ง ทำใจ อ่าน ได้ ปลาย ปี _ รัก พี่ เป้`
- norm: `ได้ มา ตั้งแต่ ปี ที่ แล้ว _ แต่ ยัง ไม่ กล้า เปิด อ่าน ต่อ _ ค้าง อยู่ ตรง ที่ เจย์ ไม่ ล็อกเอาต์ ไลน์ _ รู้สึก กลัว กับ สิ่ง ที่ จะ เกิด ขึ้น _ รู้ ทั้ง รู้ ว่า เป็น แค่ นิยาย _ แต่ สำหรับ เรา _ เรา อ่าน แล้ว รู้สึก ว่า พวก เขา เป็น ส่วน หนึ่ง ใน ชีวิต เรา _ # เจย์ ไหน _ เหมือน กับ _ # กุเชอร์รี่ _ ได้ มา เป็น ปี _ เพิ่ง ทำใจ อ่าน ได้ ปลาย ปี _ รัก พี่ เป้`
- pred: `ได้มา มาตั้งแต่ ตั้งแต่ ปี ที่ still _ แต่ ยัง ไม่ กลัว open อ่าน ต่อ _ ค้างอยู่ stay ที่ ที่ Jay ไม่ logout LINE _ feel afraid กับ สิ่งที่ ที่ จะ เกิดขึ้น เกิด _ รู้สึก ทั้งly รู้สึก ที่ เป็น เพียง นิยาย _ แต่ สำหรับ เรา _ เรา อ่าน still feel ที่ คน เขา เป็น ส่วนของ 1 ใน ชีวิต เรา _ # Jay ไหน _ like กับ _ # กูเชียร์รี่ _ ได้มา มาตั้งแต่ เป็น ปี _ just make peace อ่าน ได้มา ปลาย ปี _ รัก พี่ พี่เป้`
- categories: `{'over_normalized': 35, 'wrong_correction': 1}`

### row 3403 (ja), errors=34

- raw: `お久しぶり という か 、 ここ 数 ヶ月 は 鬱 の 手前 と いう か 、 片 脚 を 突っ込ん で い ます 。 そんな 感じ で メンタル が バキバキ だっ た の です が 、 なんとか 持ち直せ そう な 兆し が 出 て き まし た か ね ？ 油断 は 出来 ませ ん が … 。`
- norm: `お久しぶり という か 、 ここ 数 ヶ月 は 鬱 の 手前 と いう か 、 片足  を 突っ込ん で い ます 。 そんな 感じ で メンタル が バキバキ だっ た の です が 、 なんとか 持ち直せ そう な 兆し が 出 て き まし た か ね ？ 油断 は 出来 ませ ん が … 。`
- pred: `長期間会わなかった という か 、 この いくつか か月 -have うつ の 前 そして と言います か 、 一方 足 足 突っ込んで で いる です 。 そんなに 状態 で 精神 が 切れ切れ 壊れparalleled tributes の です が 、 なんとしても 回復する 状態 な 兆し が 現われ て ヶ月 改善 tributes か ね 疑問符 予想 -have 出classed 出来 ん が …… 。`
- categories: `{'over_normalized': 32, 'wrong_correction': 2}`

### row 7344 (th), errors=34

- raw: `สารภาพ ได้ ไหม ว่า แม้ ใจ อยาก ให้ ไป ต่อ แต่ กลัว เอ็มเน็ต จะ ผี ใส่ ลูก ให้ ต้อง ตาม จิก ตาม ด่า ตาม ลุ้น คือ มัน ไม่ สนุก เลย เว้ย เพราะ ไม่ ใช่ ลูกรัก ไม่ ใช่ ตัวเต็ง เขา จะ ทำ อะไร กับ หนู ก็ ได้ _ วันนี้ ผล จะ ออก มา เป็น ไง ก็ ไม่ เสียใจ _ แต่ ถ้า น้อง ได้ ไป ต่อ เตรียมตัว ไว้ แม่ๆ ว่า มัน จะ มี แน่นอน _ #กองทัพ พีค _ #KongthapPeak _ #PRODUCEX101`
- norm: `สารภาพ ได้ ไหม ว่า แม้ ใจ อยาก ให้ ไป ต่อ แต่ กลัว เอ็มเน็ต จะ ผี ใส่ ลูก ให้ ต้อง ตาม จิก ตาม ด่า ตาม ลุ้น คือ มัน ไม่ สนุก เลย เว้ย เพราะ ไม่ ใช่ ลูกรัก ไม่ ใช่ ตัวเต็ง เขา จะ ทำ อะไร กับ หนู ก็ ได้ _ วันนี้ ผล จะ ออก มา เป็น อย่างไร ก็ ไม่ เสียใจ _ แต่ ถ้า น้อง ได้ ไป ต่อ เตรียมตัว ไว้ แม่ๆ ว่า มัน จะ มี แน่นอน _ #กองทัพ พีก _ #KongthapPeak _ #PRODUCEX101`
- pred: `สารภาพ ได้ ไม่ ว่า แม้ ใจ อยาก ให้ go continue แต่ afraid มิวเน็ต จะ ปี ใส่ son ให้ must follow จิก follow ด่า follow ลุ้น because มัน ไม่ fun เลย ว่า because ไม่ จริง ลูกรัก ไม่ จริง ตัวแทน เขา จะ ทำ อะไร กับ เด็ก ก็ ได้ จริง วันนี้ ผล จะ ออกมา มา เป็น ไง ก็ ไม่ sad จริง แต่ แต่ น้อง ได้ go continue prepare ไว้ แม่ๆ ว่า มัน จะ มี แน่นอน จริง #กองทัพ Peak จริง #กองทัพ พีค จริง # ProduceX101`
- categories: `{'over_normalized': 32, 'missed_change': 1, 'wrong_correction': 1}`

### row 3675 (ko), errors=32

- raw: `출산율 대책 그냥 네델란드 처럼 노인들 안락사 허락해주면 됨 다만 가족들이 압박할 가능성 있기 때문에 가족과 떨어져서 살게 해달라는 사람이 있으면 그렇게 지원해주고 부양할 노인들만 줄어들면 지금 출산율로도 충분히 경쟁력있음 쏠로도 늙어죽는 도태된 자들은 늙어서도 계속 일하게 세금 팍팍 늘려버리고`
- norm: `출산율 대책 그냥 네델란드 처럼 노인들 안락사 허락해주면 됨 다만 가족들이 압박할 가능성 있기 때문에 가족과 떨어져서 살게 해달라는 사람이 있으면 그렇게 지원해주고 부양할 노인들만 줄어들면 지금 출산율로도 충분히 경쟁력있음 쏠로도 늙어죽는 도태된 자들은 늙어서도 계속 일하게 세금 팍팍 늘려버리고`
- pred: `출산율 대책 simply 나etherlands 一样 노인们 장기요양 서비스 허가해 주면 되рма 다만 가족들이 압력받을 가능성이 있음 because 가족이 分开 살아 달라주는 인 있으면 那样 지원해 주고 양육할 노인们仅 줄어들면 현재 출산율로 충분하다 경쟁력있음 싱글도 노년에 죽는 퇴역 들 elderly also continuously 일하게 tax rapidly 늘려주고`
- categories: `{'over_normalized': 32}`

### row 3472 (ja), errors=30

- raw: `拙者 は 薬物 の 成分 名 と 商品名 を ごっちゃ に 書く やつ を 許さ ない 侍 … 参り ます … … ！！( 他 の 学生 の レポート を 読ん で 質問 を 書く やつ を やっ て い ます )`
- norm: `拙者 は 薬物 の 成分 名 と 商品名 を ごっちゃ に 書く やつ を 許さ ない 侍 … 参り ます … … ！！( 他 の 学生 の レポート を 読ん で 質問 を 書く やつ を やっ て い ます 。 )`
- pred: `侍 は 薬物 のを 成分数 名前 そして 製品名 を 混同 にgradation 書くやつ 者 を 許す ない 侍 …… 参ります です …… …… ！！( 他の のを 学者 のを 報告書 を 読み で 問題 を 書くやつ 者 を やらない で いない です ）`
- categories: `{'over_normalized': 29, 'wrong_correction': 1}`

### row 3534 (ja), errors=27

- raw: `あと 『 note 』 に も かなり 救わ れ て い ます 。 「 スキ 」 で 反応 し て もらえる の は 嬉しい です ね ！ コメント を つけ て 引用 RT し て くれ たら 爆速 で RT する よう に し て い ます 。`
- norm: `あと 『 note 』 に も かなり 救わ れ て い ます 。 「 スキ 」 で 反応 し て もらえる の は 嬉しい です ね ！ コメント を つけ て 引用 リツイート し て くれ たら 爆速 で リツイート する よう に し て い ます 。`
- pred: `その後 「 ノート ） に 多くの かなり 助けられる 救援 て Influence ます 。 「 いいね ） で 応答 して て 受けられる です は 喜び ですね ですね ですね コメントを を 付け て 引用 リツイート して て くれたら then 高速 で リツイート する やりように に して て Influence ます 。`
- categories: `{'over_normalized': 27}`

### row 3397 (ja), errors=26

- raw: `ありがたい こと に 、 この ご時世 で 在宅ワーク で 仕事 を やら せ て もらっ て おり ます が 、 次々 と 舞い込む 仕事 に 消耗 … 。 慢性 的 な キャパオーバー な ので 、 どうにか 打開 し たい です 。`
- norm: `ありがたい こと に 、 この ご時世 で 在宅ワーク で 仕事 を やら せ て もらっ て おり ます が 、 次々 と 舞い込む 仕事 に 消耗 し て い ます … 。 慢性 的 な キャパシティ オーバー な ので 、 どうにか 打開 し たい です 。`
- pred: `幸い ことに に 、 その 時代 で リモートワーク で 仕事 ework 仕事を させ で もらっ で 存在 です が 、 次々と そして 入る 仕事 に 仕事 …… 。 慢性的 てき なaminer キャパシティオーバー なaminer で 、 なんとか 克服 対策 たい です 。`
- categories: `{'over_normalized': 24, 'wrong_correction': 2}`

### row 7291 (th), errors=25

- raw: `คน ละ เรื่อง กัน เลย _ ยุค พรรค เพื่อไทย ( ไทยรักไทย ) _ ประเทศ ไทย เจริญ ที่สุด _ เศรษฐกิจ ดี ที่สุด _ ประชาชน มี ความสุข และ ภาคภูมิใจ ใน ประเทศ ที่สุด อิสระ ที่สุ _ นายก เอาใจ ประชาชน ที่สุด _ ตั้งแต่ เริ่ม เป็น รัฐบาล แรก กัน เลย _ ที่ ต้อง ซ่อม ประเทศ จาก ยุค ต้มยำกุ้ง นายก ใน ดวงใจ _ ดร. ทักษิน ชินวัตร _ ได้ สร้าง ยุค รุ่งเรือง ที่สุด`
- norm: `คน ละ เรื่อง กัน เลย _ ยุค พรรค เพื่อไทย ( ไทยรักไทย ) _ ประเทศ ไทย เจริญ ที่สุด _ เศรษฐกิจ ดี ที่สุด _ ประชาชน มี ความสุข และ ภาคภูมิใจ ใน ประเทศ ที่สุด อิสระ ที่สุด _ นายกรัฐมนตรี เอาใจ ประชาชน ที่สุด _ ตั้งแต่ เริ่ม เป็น รัฐบาล แรก กัน เลย _ ที่ ต้อง ซ่อม ประเทศ จาก ยุค ต้มยำกุ้ง นายกรัฐมนตรี ใน ดวงใจ _ ดร. ทักษิน ชินวัตร _ ได้ สร้าง ยุค รุ่งเรือง ที่สุด`
- pred: `คน ไม่ เรื่อง กันเลย แล้ว _ ยุค party เพื่อไทย ( เพื่อไทย ) _ ประเทศ ไทย THRIVETH extreme _ เศรษฐกิจ ดี extreme _ ประชาชน มี ความ สุข และ ภาคภูมิใจ ใน ประเทศ extreme อิสระ สุด _ นายกฯ pay attention ประชาชน extreme _ ตั้งแต่ เริ่ม เป็น รัฐบาล แรก กันเลย แล้ว _ ที่ ต้อง ซ่อมแซม ประเทศ จาก ยุค ยำกุ้ง นายกฯ ใน ใจ _ ดร ทักษิณ ชินวัตร _ ได้ create ยุค prosperous extreme`
- categories: `{'over_normalized': 22, 'wrong_correction': 3}`
