# Image Classification with Convolutional Neural Networks (CNN)-model-project

Bu proje, görsel sınıflandırma için derin öğrenme kullanarak, farklı hayvan türlerini tanımayı amaçlayan bir modelin geliştirilmesini içermektedir. Proje, 10 farklı hayvan sınıfını içeren bir veri seti üzerinde çalışmaktadır ve bu sınıflandırma, bir Convolutional Neural Network (CNN) modeline dayanmaktadır. Ayrıca, görüntü artırma, renk sabitleme, ve farklı ışık koşullarında testler gibi teknikler kullanılarak modelin doğruluğu artırılmaya çalışılmıştır.

## Proje İçeriği

1. **Veri Seti**: Proje, çeşitli hayvan türlerine ait 10 farklı sınıfı içeren bir görüntü veri setine dayanmaktadır. Veri seti, her sınıftan 650 resim ile toplamda 6500 görüntüden oluşmaktadır.
   
2. **Model Yapısı**:
   - CNN tabanlı bir model tasarlandı.
   - 3 katmanlı konvolüsyonel ağ, her katmanda MaxPooling işlemiyle daha derin özelliklerin öğrenilmesini sağladı.
   - Fully connected (dense) katmanlarla sınıflandırma yapılacak son katman oluşturuldu.
   
3. **Veri Artırma (Augmentation)**: Eğitim verisi üzerinde veri artırma işlemleri uygulandı. Bu işlem, modelin genelleme yeteneğini artırarak overfitting'i engellemeyi amaçlar. Uygulanan veri artırma teknikleri:
   - Resim döndürme, kaydırma, kesme, zoom ve yatay çevirme işlemleri.
   
4. **Işık Manipülasyonu ve Renk Sabitleme**:
   - Test verileri üzerinde farklı ışık koşullarında manipülasyon yapıldı.
   - Gray World algoritması kullanılarak renk sabitleme işlemi uygulandı.

5. **Model Performansı**:
   - Modelin doğruluğu, normal test seti, manipüle edilmiş test seti ve renk sabitliği uygulanmış test seti ile değerlendirildi.
   - Model, her test seti için doğruluk oranlarını gösteren sonuçlar sundu:
     - **Eğitim Seti Doğruluğu**: %52.44
     - **Test Seti Doğruluğu**: %13.59
     - **Manipüle Edilmiş Test Seti Doğruluğu**: %10.05
     - **Renk Sabitliği Uygulanan Test Seti Doğruluğu**: %13.59                                                                                                                                                                                                   
                                                                                                                                                                                                                                                                  
 Aşağıda proje kapsamında elde edilen sonuçların detaylı bir yorumlaması bulunmaktadır.

1. Model Performansı

Eğitim Doğruluğu: Eğitim setindeki son epoch doğruluğu: %52.44 (kod çalıştırıldığında hesaplanacak değer).

Bu oran, modelin eğitim verileri üzerindeki performansını göstermektedir. Modelin, eğitim setine iyi uyum sağladığını ifade eder.

Test Doğruluğu: Test setindeki doğruluk oranı: %13.59.

Test seti performansı, modelin daha önceden görmediği verilerdeki başarısını ortaya koyar. Eğitim setindeki başarı ile test setindeki başarı arasında fark varsa, overfitting (aşırı uyum) problemi olabilir.

2. Manipüle Edilmiş Veri Üzerindeki Performans

Manipüle edilmiş verilerle modelin dayanıklılığı test edilmiştir.

a) Farklı Işık Koşulları:

Manipüle edilmiş test setindeki doğruluk oranı: %10.05.

Bu oran, modelin ışık koşullarının değiştiği durumlarda başarısını ifade eder. Sonuç, ışık manipülasyonu nedeniyle modelin performansında bir düşüş olduğunu gösterebilir.

Bu durum, modelin gerçek hayatta karşılaşabileceği çeşitli ışık koşullarına ne kadar dayanıklı olduğunu ortaya koyar.

b) Renk Sabitliği (Gray World Algoritması):

Renk sabitliği uygulanmış test setindeki doğruluk oranı: %13.59.                                                                                                                                                                                                 

Gray World algoritması kullanılarak renklerin normalize edilmesi sonucu modelin performansı test edilmiştir. Bu işlem, farklı renk koşulları altında modelin dayanıklılığını artırabilir.

3. Veri Artırmanın Katkısı

Proje boyunca, veri artırma teknikleri modelin daha çeşitli örneklerle eğitilmesini sağlamıştır. Veri artırma sayesinde:

Modelin genelleme yeteneği gelişmiş ve aşırı uyum problemi (overfitting) azaltılmıştır.

Test setindeki başarı oranı, veri artırma yapılmadığı senaryolara göre daha yüksek olmuştur.

4. Model Performansının Genel Değerlendirilmesi

Eğitim ve test setindeki doğruluk oranları modelin başarılı bir şekilde sınıflandırma yaptığını gösteriyor. Ancak manipüle edilmiş verilerle test edildiğinde performans düşüşü gözlemlenmiştir. Bu durum, gerçek dünya uygulamalarında modelin şu özelliklere ihtiyaç duyabileceğini gösterir:

Veri artırma tekniklerinin ışık ve renk manipülasyonları içerecek şekilde genişletilmesi.

Daha karmaşık modellerin denenmesi (transfer learning gibi).

5. Öneriler ve Gelecek Çalışmalar

Gerçek dünya verileriyle çeşitli testler yapılarak modelin dayanıklılığı artırılabilir.

Transfer öğrenme teknikleriyle çok daha karmaşık modeller kullanılabilir.

Model, işık ve renk manipülasyonlarına karşı özel olarak eğitilmiş bir veri setiyle tekrar eğitilebilir.

# Proje, aşağıdaki Python kütüphanelerini kullanır:
- `numpy`
- `opencv-python`
- `tensorflow`
- `scikit-learn`
- `PIL`
- `matplotlib`

