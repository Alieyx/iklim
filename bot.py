import discord  # Discord kütüphanesi
from discord.ext import commands  # Komut sistemi
import os  # Dosya işlemleri
import numpy as np
from PIL import Image, ImageOps
import aiosqlite  # Puanlama sistemi için hafif veritabanı

# Model yükleme kütüphanesi
try:
    from keras.models import load_model
except:
    from tensorflow.keras.models import load_model

# Bot izinlerini ayarla
intents = discord.Intents.default()
intents.messages = True
intents.guilds = True
intents.members = True  # Rol ve üye kontrolleri için şart
intents.message_content = True

# Botu oluştur
bot = commands.Bot(command_prefix="!", intents=intents, help_command=None)

# Görsellerin kaydedileceği klasör
IMAGE_DIR = "images"
os.makedirs(IMAGE_DIR, exist_ok=True)

MODEL_PATH = "keras_model.h5"
LABELS_PATH = "labels.txt"

model = None
labels = []

# --- AYARLAR ---
YETKILI_ROL_ID = 1525564302788919587  # Sadece bu role sahip olanlar puan verebilir

MARKET_URUNLERI = {
    "fidan": {"isim": "🌱 Fidan Dikici", "id": 1525562756604891277, "fiyat": 5},
    "doğa": {"isim": "💧 Doğa Koruyucu", "id": 1525562990836060362, "fiyat": 10},
    "çöp": {"isim": "🗑️ Çöp Avcısı", "id": 1525563056288043070, "fiyat": 15},
    "eko": {"isim": "🌍 Eko-Savaşçı", "id": 1525563135707058256, "fiyat": 25},
    "bakan": {"isim": "👑 İklim Bakanı", "id": 1525563218863460444, "fiyat": 50}
}

# Yapay zekayı yükleyen fonksiyon
def load_teachable_machine():
    global model, labels
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
        print("❌ HATA: Model veya etiket dosyası bulunamadı!")
        return False
    
    model = load_model(MODEL_PATH, compile=False)
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = [line.strip().split(" ", 1)[-1].strip() for line in f if line.strip()]
    
    print("✓ MODEL BAŞARIYLA YÜKLENDİ! SİSTEM HAZIR.")
    return True

# Sınıflandırma yapan fonksiyon
def get_class(file_path):
    image = Image.open(file_path).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    
    image_array = np.asarray(image, dtype=np.float32)
    normalized_image_array = (image_array / 127.5) - 1
    
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    prediction = model.predict(data)
    index = np.argmax(prediction)
    return labels[index], float(prediction[0][index])

# --- VERİTABANI İŞLEMLERİ ---
async def puan_ekle_miktar(user_id: int, miktar: int):
    async with aiosqlite.connect("puanlar.db") as db:
        await db.execute("""
            INSERT INTO kullanicilar (user_id, uint_puan) 
            VALUES (?, ?) 
            ON CONFLICT(user_id) 
            DO UPDATE SET uint_puan = uint_puan + ?
        """, (str(user_id), miktar, miktar))
        await db.commit()

async def puan_dus(user_id: int, miktar: int):
    async with aiosqlite.connect("puanlar.db") as db:
        await db.execute("UPDATE kullanicilar SET uint_puan = uint_puan - ? WHERE user_id = ?", (miktar, str(user_id)))
        await db.commit()

async def puan_getir(user_id: int):
    async with aiosqlite.connect("puanlar.db") as db:
        async with db.execute("SELECT uint_puan FROM kullanicilar WHERE user_id = ?", (str(user_id),)) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else 0


# --- KOMUTLAR ---

@bot.command()
async def check(ctx):
    if not ctx.message.attachments:
        await ctx.send("⚠️ Görsel yüklemeyi unuttun!")
        return
    
    for attachment in ctx.message.attachments:
        file_name = attachment.filename
        file_path = os.path.join(IMAGE_DIR, file_name)
        
        try:
            await attachment.save(file_path)
            await ctx.send(f"✅ Görsel başarıyla kaydedildi: `{file_name}`")
            
            # Tahmini alıyoruz
            class_name, confidence_score = get_class(file_path)
            clean_class_name = class_name.lower().strip()

            await ctx.send(f"🔍 Sınıflandırma sonucu: `{class_name}`\n🔍 Güven skoru: `{confidence_score:.2f}`")

            # Çevre çıkarımları sözlüğü
            cikarimlar = {
                "copler": "🗑️",
                "agac_sulama": "💧",
                "agac_ekme": "🌱"
            }

            # Çıkarımı alıyoruz
            cikarim = cikarimlar.get(clean_class_name, "🌿 Bu çevre sınıfı için henüz özel bir çıkarım eklenmedi.")
            await ctx.send(f"🔍 Çıkarım: `{cikarim}`")

            # Eğer bilinen çevre sınıflarından biriyse +1 Puan veriyoruz
            if clean_class_name in cikarimlar:
                await puan_ekle_miktar(ctx.author.id, 1)
                yeni_puan = await puan_getir(ctx.author.id)
                await ctx.send(f"⭐ Tebrikler {ctx.author.mention}! Doğa dostu eylemin için **+1 Puan** kazandın. Toplam Puanın: **{yeni_puan}**")

            # İş bitince resmi siliyoruz diski doldurmasın
            if os.path.exists(file_path):
                os.remove(file_path)

        except Exception as e:
            await ctx.send(f"⚠️ Hata oluştu: {str(e)}")

@bot.command(name="puan")
async def puan_sorgula(ctx):
    try:
        puan = await puan_getir(ctx.author.id)
        await ctx.send(f"📊 {ctx.author.mention}, şu anki toplam puanın: **⭐ {puan}**")
    except Exception as e:
        await ctx.send(f"⚠️ Puan getirilirken hata oluştu: {str(e)}")

@bot.command(name="puanlar")
async def top_puanlar(ctx):
    try:
        async with aiosqlite.connect("puanlar.db") as db:
            async with db.execute("SELECT user_id, uint_puan FROM kullanicilar ORDER BY uint_puan DESC LIMIT 10") as cursor:
                rows = await cursor.fetchall()
                
        if not rows:
            await ctx.send("🤖 Henüz puan alan kimse yok!")
            return
        
        embed = discord.Embed(title="🏆 En Çevreci İlk 10 Üye (Top 10)", color=discord.Color.gold())
        liste_metni = ""
        for sira, row in enumerate(rows, start=1):
            kullanici_id = int(row[0])
            puan = row[1]
            kullanici = ctx.guild.get_member(kullanici_id)
            kullanici_adi = kullanici.mention if kullanici else f"Kullanıcı ({kullanici_id})"
            emoji = "🥇" if sira == 1 else "🥈" if sira == 2 else "🥉" if sira == 3 else f"`{sira}.`"
            liste_metni += f"{emoji} {kullanici_adi} — **{puan} Puan**\n"
            
        embed.description = liste_metni
        await ctx.send(embed=embed)
    except Exception as e:
        await ctx.send(f"⚠️ Liderlik tablosu hatası: {str(e)}")

# ⭐ YETKİLİYE ÖZEL PUAN VERME KOMUTU
@bot.command(name="puanver")
async def yetkili_puan_ver(ctx, hedef_uye: discord.Member = None, miktar: int = None):
    # Komutu kullanan kişinin yetkili rolü var mı kontrol et
    yetkili_rol = ctx.guild.get_role(YETKILI_ROL_ID)
    if yetkili_rol not in ctx.author.roles:
        await ctx.send("❌ Hata: Bu komutu kullanmak için gerekli yetkili rolüne sahip değilsiniz!")
        return

    # Argümanların eksik olup olmadığını kontrol et
    if hedef_uye is None or miktar is None:
        await ctx.send("⚠️ Yanlış kullanım! Doğru format: `!puanver @kullanıcı [miktar]`\nÖrnek: `!puanver @Ragnar 10`")
        return

    if miktar <= 0:
        await ctx.send("⚠️ Verilecek puan miktarı 0'dan büyük olmalıdır!")
        return

    try:
        await puan_ekle_miktar(hedef_uye.id, miktar)
        guncel_puan = await puan_getir(hedef_uye.id)
        await ctx.send(f"✅ Başarılı! {ctx.author.mention} isimli yetkili, {hedef_uye.mention} kullanıcısına **{miktar} Puan** verdi. Kullanıcının Yeni Puanı: **⭐ {guncel_puan}**")
    except Exception as e:
        await ctx.send(f"⚠️ Puan verilirken bir hata oluştu: {str(e)}")

@bot.command(name="market")
async def marketi_goster(ctx):
    embed = discord.Embed(title="🛒 Çevreci Rütbe Market Menüsü", color=discord.Color.green())
    embed.description = "Topladığın çevre puanlarıyla aşağıdan kendine efsane rütbeler satın alabilirsin!\n\n**Satın almak için:** `!satınal [rütbe_adı]`"
    
    for anahtar, veri in MARKET_URUNLERI.items():
        embed.add_field(
            name=veri["isim"], 
            value=f"Fiyat: **⭐ {veri['fiyat']} Puan**\nKomut: `!satınal {anahtar}`", 
            inline=False
        )
    await ctx.send(embed=embed)

@bot.command(name="satınal")
async def urun_satinal(ctx, urun_adi: str = None):
    if not urun_adi:
        await ctx.send("⚠️ Lütfen satın almak istediğin rütbe adını yaz! Örnek: `!satınal fidan`")
        return
        
    urun_adi = urun_adi.lower().strip()
    if urun_adi not in MARKET_URUNLERI:
        await ctx.send("❌ Marketimizde böyle bir rütbe bulunamadı. `!market` yazarak ürünlere göz atabilirsin.")
        return
        
    secilen_urun = MARKET_URUNLERI[urun_adi]
    kullanici_puan = await puan_getir(ctx.author.id)
    
    if kullanici_puan < secilen_urun["fiyat"]:
        await ctx.send(f"❌ Yetersiz Puan! Bu rütbe için **{secilen_urun['fiyat']} Puan** gerekiyor. Sende olan: **{kullanici_puan} Puan**.")
        return
        
    rol = ctx.guild.get_role(secilen_urun["id"])
    if not rol:
        await ctx.send("❌ Hata: Bu rol sunucuda bulunamadı veya silinmiş. Lütfen sunucu sahibine danışın.")
        return
        
    if rol in ctx.author.roles:
        await ctx.send(f"⚠️ Zaten bu rütbeye (`{secilen_urun['isim']}`) sahipsin!")
        return
        
    try:
        await ctx.author.add_roles(rol)
        await puan_dus(ctx.author.id, secilen_urun["fiyat"])
        kalan_puan = await puan_getir(ctx.author.id)
        await ctx.send(f"🎉 Tebrikler {ctx.author.mention}! **{secilen_urun['isim']}** rütbesini başarıyla satın aldın! Kalan Puanın: **⭐ {kalan_puan}**")
    except discord.Forbidden:
        await ctx.send("❌ Hata: Botun rolleri yönetme yetkisi yok ya da botun rolü satın alınmak istenen rolden daha aşağıda!")
    except Exception as e:
        await ctx.send(f"⚠️ Satın alma sırasında beklenmeyen bir hata oluştu: {str(e)}")

@bot.command(name="cmd")
async def komutlar_listesi(ctx):
    embed = discord.Embed(title="🌿 İklim Botu Komut Menüsü", color=discord.Color.blue())
    embed.add_field(name="📸 !check", value="Çevre/İklim fotoğrafı yükleyip yapay zekaya analiz ettirir ve puan kazandırır.", inline=False)
    embed.add_field(name="📊 !puan", value="Kendi güncel puanınızı gösterir.", inline=True)
    embed.add_field(name="🏆 !puanlar", value="Sunucudaki Top 10 liderlik tablosunu gösterir.", inline=True)
    embed.add_field(name="🛒 !market", value="Puanlarınla rütbe alabileceğin mağazayı açar.", inline=True)
    embed.add_field(name="🛍️ !satınal [ad]", value="Belirtilen rütbeyi puan karşılığında satın alır.", inline=True)
    embed.add_field(name="🛠️ !puanver [@üye] [miktar]", value="*(Sadece Yetkililer)* Belirtilen üyeye elden puan ekler.", inline=False)
    await ctx.send(embed=embed)

@bot.event
async def on_ready():
    print(f'🤖 {bot.user.name} aktif!')
    try:
        async with aiosqlite.connect("puanlar.db") as db:
            await db.execute("CREATE TABLE IF NOT EXISTS kullanicilar (user_id TEXT PRIMARY KEY, uint_puan INTEGER DEFAULT 0)")
            await db.commit()
        print("✓ Veritabanı başarıyla bağlandı.")
    except Exception as e:
        print(f"❌ Veritabanı oluşturma hatası: {str(e)}")
    load_teachable_machine()

# ⚠️ BURAYA PORTALDEKİ EN GÜNCEL TOKENİNİ YAPIŞTIR
TOKEN = "TOKEN HERE"
bot.run(TOKEN)