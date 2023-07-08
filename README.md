# FakeFaceDetector

## Özet
Bu çalışmada, sahte yüzlerin tespiti için üç farklı CNN modeli sunulmaktadır: MobileNetV2, DenseNet201 ve önerilen özgün CNN modeli. Bu modeller, sahte yüzlerin tespiti için yüksek doğruluk ve güvenilirlik sağlamak amacıyla tasarlanmıştır. Öncelikle, eğitim süreci için büyük çaplı bir veri seti toplanmış ve etiketlenmiştir. Bu veri seti, gerçek ve sahte yüz görüntülerini içeren geniş bir çeşitlilik sunmaktadır. Ardından, eğitim süreci için MobileNetV2 ve DenseNet201 gibi önceden eğitilmiş ağırlıklar kullanılmıştır. Bu modeller, transfer öğrenme yöntemleriyle geliştirilmiştir ve sahte yüzlerin tespiti için etkili bir performans sergilemektedir. Ayrıca, önerilen özgün CNN modeli de geliştirilmiştir. Bu model, sahte yüzlerin tespiti için özel olarak tasarlanmış ve optimize edilmiştir. Kendi CNN modelimiz, veri setindeki özellikleri daha iyi öğrenmek ve daha hassas bir tespit sağlamak amacıyla eğitim sürecinde kullanılmıştır. Eğitim tamamlandıktan sonra, modeller test veri seti üzerinde değerlendirilmiştir. Elde edilen sonuçlar, her üç modelin de sahte yüzlerin tespiti konusunda yüksek performans sergilediğini göstermiştir. Modeller, gerçek yüzler ile sahte yüzler arasındaki farklılıkları belirlemek için farklı özelliklerin öğrenilmesini sağlamaktadır. Sonuç olarak, bu çalışmada sunulan MobileNetV2, DenseNet201 ve özgün CNN modeli, sahte yüzlerin tespiti konusunda yüksek doğruluk ve güvenilirlik sağlama potansiyeline sahiptir. Bu modeller, sahte yüzlerin otomatik olarak tespit edilmesiyle kişisel güvenliği artırabilir ve kötü niyetli faaliyetleri önleme konusunda önemli bir rol oynayabilir. Gelecekteki çalışmalar, farklı veri setleri üzerinde modellerin performansının daha da değerlendirilmesini ve yeni yöntemlerin keşfedilmesini içermelidir.

## 1. Giriş
Dijital görüntü manipülasyonu teknolojilerinin hızla ilerlemesiyle birlikte, sahte yüzlerin oluşturulması ve kullanılması potansiyel riskleri artırmıştır. Bu riskler, kişisel verilerin kötüye kullanımı, itibar kaybı, dolandırıcılık ve diğer kötü niyetli eylemler gibi bir dizi sorunu beraberinde getirmektedir. Bu nedenle, sahte yüzlerin tespit edilmesi ve doğrulanması, bireylerin güvenliğini sağlamak ve yanıltıcı uygulamalara karşı korumak için büyük önem taşımaktadır. 

Geleneksel sahte yüz tespiti yöntemleri, bir dizi el işareti ve görsel ipucu temelinde çalışırken, bu yöntemlerin sınırlamaları bulunmaktadır. Örneğin, manuel denetim gerektirebilir, hatalı sonuçlara neden olabilir veya yeni ve sofistike sahte yüzlerin tanımlanmasını zorlaştırabilir. Bu zorluklarla başa çıkmak için, derin öğrenme teknikleri ve özellikle Evrişimli Sinir Ağları (CNN'ler), sahte yüz tespiti için etkili bir yaklaşım sunmaktadır. 

Son yıllarda, sahte yüz tespiti alanında birçok araştırma yapılmıştır. Önceki çalışmalarda, farklı CNN modelleri ve öğrenme yöntemleri kullanılarak sahte yüzlerin tespiti ve doğrulanması üzerinde çalışılmıştır. Örneğin, ResNet, VGGNet ve Inception gibi popüler CNN modelleri sahte yüz tespiti için başarıyla kullanılmıştır. Ayrıca, transfer öğrenme teknikleri de kullanılarak, önceden eğitilmiş ağırlıkların yeni sahte yüz veri setlerine uygulanmasıyla modellerin performansı artırılmıştır. Bununla birlikte, her modelin kendine özgü avantajları ve dezavantajları vardır. MobileNetV2, hafif yapısı ve düşük hesaplama maliyeti nedeniyle sahte yüz tespiti için tercih edilebilir. DenseNet201 ise derin ağ yapısı sayesinde daha karmaşık özellikleri öğrenme yeteneğiyle öne çıkmaktadır. Özgün CNN modeli ise veri setine özgü özellikleri daha iyi öğrenebilme ve hassas bir tespit yapabilme potansiyeline sahiptir. Bu çalışmada, sahte yüzlerin tespiti için geliştirilen MobileNetV2, DenseNet201 ve özgün CNN modeli, önceki çalışmalardan faydalanarak ve veri setine özgü özellikleri dikkate alarak tasarlanmıştır. Amacımız, bu modellerin sahte yüzleri tespit etme ve doğrulama konusundaki performansını değerlendirmek ve sahte yüzlerin tespiti alanında yeni bir adım atarak güvenliği sağlamaktır.

## 2. Metadoloji
Önerilen yöntem, veri setinin toplanması, kullanılacak araçların ve dilin belirlenmesi, veri setindeki görüntülerin ön işlemesinin yapılması, model mimarisinin oluşturulması, modelin derlenmesi, eğitiminin ve doğrulamanın gerçekleştirilmesini içermektedir.

## 2.1. Veri seti
Bu çalışmadaki veri seti 74,184 adet gerçek-sahte görüntüye sahip insan yüzlerinden oluşmaktadır. Bu görüntülerden 60,000 tanesi fake insan yüzlerini içerirken 14.184 tanesi gerçek insan yüzlerini içermektedir. Eğitim sürecinin tekdüze olmaması ve modelin daha iyi sonuçlar vermesi için çeşitli kaynaklar taranmış ve çeşitli veri setleri ile görüntüler kullanılmıştır. Bu çeşitlilik, farklı görüntü tiplerini ve sahte yüzlerin farklı varyasyonlarını içeren geniş bir veri setinin oluşturulmasına olanak sağlamıştır. Böylece, model farklı kaynaklardan gelen veriler üzerinde eğitilerek, daha genel ve kapsayıcı bir öğrenme gerçekleştirebilmiştir. Bu sayede, modelin farklı tipteki sahte yüzleri tespit etme ve doğrulama yeteneği geliştirilmiş ve gerçek hayattaki senaryolara daha iyi uyum sağlaması hedeflenmiştir. Veri seti; Kaggle, generated.photos, Reddit, OpenSea, MyModernMet, Pinterest gibi siteler ve Google Görsellerde bulunan diğer kaynaklardan derlenmiştir. Bu çeşitli kaynaklar, geniş bir kapsamda gerçek ve sahte insan yüzlerini içermektedir. Kaggle, veri bilimi topluluğunun paylaştığı geniş bir veri seti deposu olduğu için çeşitli gerçek ve sahte insan yüzü görüntülerini içermektedir. generated.photos ise sahte yüzlerin üretildiği bir platformdur ve sahte yüz verilerini sağlamaktadır. Ayrıca, Google Görseller'de geniş bir yüz veri tabanı bulunmaktadır ve veri setinin oluşturulmasında buradaki kaynaklardan da yararlanılmıştır. 

Veri seti, eğitim, doğrulama ve test için ayrılmıştır. Eğitim için veri setinin %70'i kullanılmıştır, böylece modelin temel öğrenme süreci bu büyük veri alt kümesi üzerinde gerçekleştirilmiştir. Doğrulama için veri setinin %20'si ayrılmış ve modelin performansının izlenmesi ve hiperparametre ayarlaması için kullanılmıştır. Geriye kalan %10'luk veri seti ise test için ayrılmış ve modelin nihai performansını değerlendirmek amacıyla kullanılmıştır. Bu ayrım, modelin eğitim sürecinde aşırı uyma (overfitting) durumunu kontrol etmek ve genellemesini değerlendirmek için önemli bir adımdır.

## 2.2. Kullanılan Dil Ve Araçlar
Bu projede Python programlama dili kullanılmıştır. Python, geniş bir kütüphane ekosistemi ve kolay okunabilir syntax yapısıyla makine öğrenimi ve derin öğrenme gibi görevler için popüler bir tercihtir. 

Proje geliştirme aşamasında Jupyter Notebook kullanılmıştır. Jupyter Notebook, interaktif bir kod çalışma ortamı sunarak kodu adım adım çalıştırma, sonuçları görselleştirme ve belgelendirme yapma imkanı sağlamaktadır. Bu özellikleri sayesinde modelin oluşturulması ve test edilmesi süreçlerinde kullanıcı dostu bir ortam sunmaktadır. 

TensorFlow kütüphanesi, derin öğrenme modellerinin oluşturulması, eğitimi ve değerlendirilmesi için kullanılmıştır. TensorFlow, yüksek performanslı hesaplama yetenekleri ve geniş bir derin öğrenme araç seti sunarak modelin verimli bir şekilde geliştirilmesine ve eğitilmesine yardımcı olmuştur. Ayrıca TensorFlow-GPU, GPU hızlandırma özelliği sayesinde eğitim sürecinde daha yüksek performans elde etmemizi sağlamıştır. 

Proje için ayrıca OpenCV, matplotlib ve diğer bazı kütüphaneler de kullanılmıştır. OpenCV, görüntü işleme işlevleri için kullanılan bir kütüphanedir ve görüntülerin işlenmesi, dönüşümleri ve ön işlemeleri için kullanılmıştır. Matplotlib, veri görselleştirme için kullanılan bir kütüphanedir ve modelin performansını analiz etmek ve sonuçları görselleştirmek için kullanılmıştır. Bu dil ve araçlar, projenin başarılı bir şekilde gerçekleştirilmesi için sağlam bir temel oluşturmuştur. Veri işleme, model oluşturma ve sonuç analizi gibi adımların etkin bir şekilde gerçekleştirilmesini sağlamıştır. 

## 2.3. Görüntü Formatı
Bu projede görüntü formatı olarak JPEG (Joint Photographic Experts Group) formatı kullanılmıştır. JPEG, sıkıştırma algoritmasıyla tanınan ve fotoğrafçılıkta yaygın olarak kullanılan bir format olarak bilinir. JPEG formatı, yüksek kalitede görüntülerin daha düşük dosya boyutlarıyla saklanmasına olanak tanırken, görsel detayları korumak için kayıplı sıkıştırma yöntemini kullanır. Bu sayede, veri setindeki görüntülerin işlenmesi ve modele beslenmesi için uygun bir format seçilmiştir.

## Ön İşleme
Veri ön işleme adımları kapsamında, 'fake_real' dizininde bulunan görüntüler kullanılmıştır. Projede kullanılan MobilenetV2 ve Densenet201, 224x224 piksel boyutunda görüntüler beklemektedir. Bu nedenle bu modellerin eğitim ve test sürecinde görüntüler 224x224 olarak ayarlanmıştır. Diğer yandan özgün olarak geliştirilen CNN modelinin eğitim ve test sürecinde görüntüler 256x256 olarak ayarlanmıştır. Bu şekilde, farklı modellerin gereksinimlerini karşılayacak şekilde görüntüler ön işlenmiş ve kullanıma hazır hale getirilmiştir. Bunun yanında veri seti piksel değerlerini [0, 255] aralığından [0, 1] aralığına ölçeklendirmek için data.map fonksiyonu kullanılarak ön işleme adımı gerçekleştirilmiştir. Bu adım, görüntülerin daha iyi öğrenme performansı sağlaması ve daha istikrarlı bir eğitim süreci geçirmesi için önemlidir. 

![image](https://github.com/altaysoprano/FakeFaceDetector/assets/37440249/d6299c2b-cc51-4a75-b7f3-cd8d8dcd874b)
![image](https://github.com/altaysoprano/FakeFaceDetector/assets/37440249/ddb74810-1462-418f-a637-2363946924d3)
![image](https://github.com/altaysoprano/FakeFaceDetector/assets/37440249/ecd89a79-5e1a-487a-a1a9-6eba76960ebf)
Şekil 2. Özgün CNN modeli için kullanılan görüntü örnekleri

![image](https://github.com/altaysoprano/FakeFaceDetector/assets/37440249/63ea819e-c040-454c-b15d-cdad1825a299)
![image](https://github.com/altaysoprano/FakeFaceDetector/assets/37440249/e51aea91-820a-453d-85d9-be972bc4036c)
![image](https://github.com/altaysoprano/FakeFaceDetector/assets/37440249/04e27ee1-18ca-4178-9299-56cf68050e85)
Şekil 3. MobilenetV2 için kullanılan görüntü örnekleri

![image](https://github.com/altaysoprano/FakeFaceDetector/assets/37440249/3355752f-de5d-4364-94d8-56647d96b79a)
![image](https://github.com/altaysoprano/FakeFaceDetector/assets/37440249/cfbbbbae-0e2d-44e2-bd84-05ac0af756a8)
![image](https://github.com/altaysoprano/FakeFaceDetector/assets/37440249/555a6582-8792-4dbc-a908-10b537861fb4)
Şekil 4. DenseNet201 için kullanılan görüntü örnekleri

## 2.5. Model Ağı
Daha önce de bahsettiğimiz gibi, proje için kullanılan ‘fake_real’ veri seti, oluşturduğumuz özgün bir CNN modeli ve ön-eğitimli 2 model (MobilenetV2 ve Densenet201) kullanılarak eğitilmiştir. 

## 2.6. Modellerin Eğitimi ve Doğrulanması
## 2.6.1. Önerilen Model
Önerilen özgün model, sahte yüzlerin tespiti için bir evrişimli sinir ağı (Convolutional Neural Network - CNN) kullanmaktadır. Bu model, görüntülerin boyutu, evrişim katmanları, dropout katmanları ve tam bağlantılı katmanlar gibi çeşitli bileşenlerden oluşmaktadır. 

## 2.6.1.1. Giriş Katmanı
Görüntülerin boyutu (256, 256, 3) olarak kabul edilir, yani 256 piksel genişlik, 256 piksel yükseklik ve 3 renk kanalı (RGB) bulunur. Bu katman, modele görüntülerin girişini sağlar. 
## 2.6.1.2. Evrişim Katmanları
İlk evrişim katmanı, 16 adet (3, 3) boyutunda filtre kullanır ve ReLU aktivasyon fonksiyonu ile aktive edilir. Ardından maksimum havuzlama (MaxPooling) katmanı gelir, boyutu varsayılan olarak belirlenir. İkinci evrişim katmanı, 32 adet (3, 3) boyutunda filtre kullanır ve yine ReLU aktivasyon fonksiyonu ile aktive edilir. 
## 2.6.1.3. Dropout Katmanları
İlk Dropout katmanı, %40 oranında nöronları rastgele atarak aşırı uyumu önlemek için kullanılır. İkinci Dropout katmanı da aynı şekilde %40 oranında nöronları rastgele atar. 
## 2.6.1.4. Düzleştirme Katmanı
Düzleştirme (Flatten) katmanı, evrişim katmanlarından elde edilen özellik haritalarını düz bir vektöre dönüştürür. Bu, tam bağlantılı (Dense) katmanlarda kullanılmak üzere verilerin düzenlenmesini sağlar. 
## 2.6.1.5. Tam Bağlantılı Katmanlar
İlk tam bağlantılı katman, 256 nöron içerir ve ReLU aktivasyon fonksiyonu ile aktive edilir. Son katman, ikili sınıflandırma yapılacağı için 1 nöron içerir ve sigmoid aktivasyon fonksiyonu kullanır. 
## 2.6.1.6. Derleme
Model, 'adam' optimizer ile derlenir. 'adam' optimizer, adaptif momentum tahmini kullanarak ağırlıkları günceller. Kayıp fonksiyonu olarak Binary Crossentropy kullanılır, çünkü bu bir ikili sınıflandırma problemini çözmektedir. Doğruluk (accuracy) metriği, modelin sınıflandırma performansını ölçmek için kullanılır. 

Algoritma 1. Özgün CNN modelinin kodları
# Kütüphanelerin import edilmesi
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

# Modelin oluşturulması
model = Sequential()
from keras import regularizers
model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.4))
model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Modelin derlenmesi 
model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])



 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 254, 254, 16)      448       
                                                                 
 max_pooling2d (MaxPooling2D  (None, 127, 127, 16)     0         
 )                                                               
                                                                 
 conv2d_1 (Conv2D)           (None, 125, 125, 32)      4640      
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 62, 62, 32)       0         
 2D)                                                             
                                                                 
 dropout (Dropout)           (None, 62, 62, 32)        0         
                                                                 
 conv2d_2 (Conv2D)           (None, 60, 60, 16)        4624      
                                                                 
 max_pooling2d_2 (MaxPooling  (None, 30, 30, 16)       0         
 2D)                                                             
                                                                 
 dropout_1 (Dropout)         (None, 30, 30, 16)        0         
                                                                 
 flatten (Flatten)           (None, 14400)             0         
                                                                 
 dense (Dense)               (None, 256)               3686656   
                                                                 
 dense_1 (Dense)             (None, 1)                 257       
                                                                 
=================================================================
Total params: 3,696,625
Trainable params: 3,696,625
Non-trainable params: 0
_________________________________________________________________

Şekil 5. Özgün modelin özeti












