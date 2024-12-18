# Image Classification with Convolutional Neural Networks (CNN)

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


Proje, aşağıdaki Python kütüphanelerini kullanır:
- `numpy`
- `opencv-python`
- `tensorflow`
- `scikit-learn`
- `PIL`
- `matplotlib`

