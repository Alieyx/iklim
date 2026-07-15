import os
import discord
from discord.ext import commands
import numpy as np
from PIL import Image, ImageOps
import aiosqlite

try:
    from keras.models import load_model
except ImportError:
    from tensorflow.keras.models import load_model

# --- SABİTLER (CONSTANTS) ---
BOT_PREFIX = "!"
YETKILI_ROL_ID = 1525564302788919587
DATABASE_FILE = "puanlar.db"
IMAGE_DIRECTORY = "images"
MODEL_FILE_PATH = "keras_model.h5"
LABELS_FILE_PATH = "labels.txt"
BOT_TOKEN = (
    "YOUR TOKEN HERE"
)

# Puan kazandıran çevre sınıfları
CEVRE_SINIFLARI = ("copler", "agac_sulama", "agac_ekme")

MARKET_URUNLERI = {
    "fidan": {
        "isim": "🌱 Fidan Dikici",
        "id": 1525562756604891277,
        "fiyat": 5
    },
    "doğa": {
        "isim": "💧 Doğa Koruyucu",
        "id": 1525562990836060362,
        "fiyat": 10
    },
    "çöp": {
        "isim": "🗑️ Çöp Avcısı",
        "id": 1525563056288043070,
        "fiyat": 15
    },
    "eko": {
        "isim": "🌍 Eko-Savaşçı",
        "id": 1525563135707058256,
        "fiyat": 25
    },
    "bakan": {
        "isim": "👑 İklim Bakanı",
        "id": 1525563218863460444,
        "fiyat": 50
    }
}

# --- GLOBAL MODEL DEĞİŞKENLERİ ---
ai_model = None
ai_labels = []

# --- BOT KURULUMU ---
bot_intents = discord.Intents.default()
bot_intents.messages = True
bot_intents.guilds = True
bot_intents.members = True
bot_intents.message_content = True

bot = commands.Bot(
    command_prefix=BOT_PREFIX,
    intents=bot_intents,
    help_command=None
)
os.makedirs(IMAGE_DIRECTORY, exist_ok=True)


# --- YARDIMCI FONKSİYONLAR VE DECORATORLER ---
def has_authority_role():
    """Yetkisinin var olup olmadığını kontrol eder."""
    async def predicate(ctx):
        authority_role = ctx.guild.get_role(YETKILI_ROL_ID)
        if authority_role not in ctx.author.roles:
            await ctx.send(
                "❌ Hata: Bu komutu kullanmak için gerekli "
                "yetkili rolüne sahip değilsiniz!"
            )
            return False
        return True
    return commands.check(predicate)


def load_machine_learning_model():
    """Model dosyalarını yükler."""
    global ai_model, ai_labels
    if not os.path.exists(MODEL_FILE_PATH) or \
            not os.path.exists(LABELS_FILE_PATH):
        print("❌ HATA: Model veya etiket dosyası bulunamadı!")
        return False

    ai_model = load_model(MODEL_FILE_PATH, compile=False)
    with open(LABELS_FILE_PATH, "r", encoding="utf-8") as file:
        ai_labels = [
            line.strip().split(" ", 1)[-1].strip()
            for line in file if line.strip()
        ]

    print("✓ MODEL BAŞARIYLA YÜKLENDİ! SİSTEM HAZIR.")
    return True


def predict_image_class(file_path):
    """Teachable Machine modelini kullanarak görseli analiz eder."""
    image = Image.open(file_path).convert("RGB")
    target_size = (224, 224)
    image = ImageOps.fit(image, target_size, Image.Resampling.LANCZOS)

    image_array = np.asarray(image, dtype=np.float32)
    normalized_image_array = (image_array / 127.5) - 1

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    prediction = ai_model.predict(data)
    highest_index = np.argmax(prediction)
    return ai_labels[highest_index], float(prediction[0][highest_index])


# --- VERİTABANI YÖNETİMİ ---
async def execute_database_query(query, parameters=()):
    """Tekrarlayan veritabanı yazma/güncelleme işlemlerini ortaklaştırır."""
    async with aiosqlite.connect(DATABASE_FILE) as connection:
        await connection.execute(query, parameters)
        await connection.commit()


async def get_user_score(user_id: int) -> int:
    """Kullanıcının veritabanındaki güncel puanını döner."""
    async with aiosqlite.connect(DATABASE_FILE) as connection:
        async with connection.execute(
            "SELECT uint_puan FROM kullanicilar WHERE user_id = ?",
            (str(user_id),)
        ) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else 0


async def add_user_score(user_id: int, amount: int):
    """Kullanıcıya puan ekler veya yoksa yeni kayıt açar."""
    query = """
        INSERT INTO kullanicilar (user_id, uint_puan)
        VALUES (?, ?)
        ON CONFLICT(user_id)
        DO UPDATE SET uint_puan = uint_puan + ?
    """
    await execute_database_query(query, (str(user_id), amount, amount))


async def set_user_score(user_id: int, target_score: int):
    """Kullanıcının puanını direkt olarak belirtilen miktara eşitler."""
    query = """
        INSERT INTO kullanicilar (user_id, uint_puan)
        VALUES (?, ?)
        ON CONFLICT(user_id)
        DO UPDATE SET uint_puan = ?
    """
    await execute_database_query(
        query, (str(user_id), target_score, target_score)
    )


async def deduct_user_score(user_id: int, amount: int):
    """Kullanıcının mevcut puanından belirtilen miktarı düşer."""
    query = (
        "UPDATE kullanicilar SET uint_puan = uint_puan - ? "
        "WHERE user_id = ?"
    )
    await execute_database_query(query, (amount, str(user_id)))


# --- KULLANICI KOMUTLARI ---
@bot.command()
async def check(ctx):
    if not ctx.message.attachments:
        await ctx.send("⚠️ Görsel yüklemeyi unuttun!")
        return

    for attachment in ctx.message.attachments:
        file_path = os.path.join(IMAGE_DIRECTORY, attachment.filename)

        try:
            await attachment.save(file_path)
            await ctx.send(f"✅ Görsel kaydedildi: `{attachment.filename}`")

            detected_class, confidence = predict_image_class(file_path)
            clean_class_name = detected_class.lower().strip()

            await ctx.send(
                f"🔍 Sınıflandırma sonucu: `{detected_class}`\n"
                f"🔍 Güven skoru: `{confidence:.2f}`"
            )

            # Eğer görsel geçerli bir çevre sınıfındaysa kullanıcıyı ödüllendir
            if clean_class_name in CEVRE_SINIFLARI:
                await add_user_score(ctx.author.id, 1)
                new_score = await get_user_score(ctx.author.id)
                await ctx.send(
                    f"⭐ Tebrikler {ctx.author.mention}! Doğa dostu "
                    f"eylemin için **+1 Puan** kazandın. "
                    f"Toplam Puanın: **{new_score}**"
                )

            if os.path.exists(file_path):
                os.remove(file_path)

        except Exception as error:
            await ctx.send(f"⚠️ Hata oluştu: {str(error)}")


@bot.command(name="puan")
async def show_user_score(ctx):
    try:
        score = await get_user_score(ctx.author.id)
        await ctx.send(
            f"📊 {ctx.author.mention}, şu anki toplam puanın: **⭐ {score}**"
        )
    except Exception as error:
        await ctx.send(f"⚠️ Puan getirilirken hata oluştu: {str(error)}")


@bot.command(name="puanlar")
async def show_leaderboard(ctx):
    try:
        query = (
            "SELECT user_id, uint_puan FROM kullanicilar "
            "ORDER BY uint_puan DESC LIMIT 10"
        )
        async with aiosqlite.connect(DATABASE_FILE) as db:
            async with db.execute(query) as cursor:
                leaderboard_rows = await cursor.fetchall()

        if not leaderboard_rows:
            await ctx.send("🤖 Henüz puan alan kimse yok!")
            return

        embed = discord.Embed(
            title="🏆 En Çevreci İlk 10 Üye (Top 10)",
            color=discord.Color.gold()
        )
        leaderboard_text = ""
        for index, row in enumerate(leaderboard_rows, start=1):
            user_id = int(row[0])
            score = row[1]
            member = ctx.guild.get_member(user_id)
            user_display = (
                member.mention if member else f"Kullanıcı ({user_id})"
            )

            rank_prefix = (
                "🥇" if index == 1
                else "🥈" if index == 2
                else "🥉" if index == 3
                else f"`{index}.`"
            )
            leaderboard_text += (
                f"{rank_prefix} {user_display} — **{score} Puan**\n"
            )

        embed.description = leaderboard_text
        await ctx.send(embed=embed)
    except Exception as error:
        await ctx.send(f"⚠️ Liderlik tablosu hatası: {str(error)}")


@bot.command(name="market")
async def show_market(ctx):
    embed = discord.Embed(
        title="🛒 Çevreci Rütbe Market Menüsü",
        color=discord.Color.green()
    )
    embed.description = (
        "Topladığın çevre puanlarıyla rütbe satın alabilirsin!\n\n"
        "**Satın almak için:** `!satınal [rütbe_adı]`"
    )

    for key, data in MARKET_URUNLERI.items():
        embed.add_field(
            name=data["isim"],
            value=f"Fiyat: **⭐ {data['fiyat']} Puan**\n"
                  f"Komut: `!satınal {key}`",
            inline=False
        )
    await ctx.send(embed=embed)


@bot.command(name="satınal")
async def purchase_rank(ctx, rank_key: str = None):
    if not rank_key:
        await ctx.send(
            "⚠️ Lütfen satın almak istediğin rütbe adını yaz! "
            "Örnek: `!satınal fidan`"
        )
        return

    rank_key = rank_key.lower().strip()
    if rank_key not in MARKET_URUNLERI:
        await ctx.send(
            "❌ Marketimizde böyle bir rütbe bulunamadı. "
            "`!market` yazarak ürünlere göz atabilirsin."
        )
        return

    selected_rank = MARKET_URUNLERI[rank_key]
    user_score = await get_user_score(ctx.author.id)

    if user_score < selected_rank["fiyat"]:
        await ctx.send(
            f"❌ Yetersiz Puan! Bu rütbe için "
            f"**{selected_rank['fiyat']} Puan** gerekiyor. "
            f"Sende olan: **{user_score} Puan**."
        )
        return

    role = ctx.guild.get_role(selected_rank["id"])
    if not role:
        await ctx.send(
            "❌ Hata: Bu rol sunucuda bulunamadı veya silinmiş."
        )
        return

    if role in ctx.author.roles:
        await ctx.send(
            f"⚠️ Zaten bu rütbeye (`{selected_rank['isim']}`) sahipsin!"
        )
        return

    try:
        await ctx.author.add_roles(role)
        await deduct_user_score(ctx.author.id, selected_rank["fiyat"])
        remaining_score = await get_user_score(ctx.author.id)
        await ctx.send(
            f"🎉 Teblikler {ctx.author.mention}! "
            f"**{selected_rank['isim']}** rütbesini başarıyla satın aldın! "
            f"Kalan Puanın: **⭐ {remaining_score}**"
        )
    except discord.Forbidden:
        await ctx.send(
            "❌ Hata: Botun rolleri yönetme yetkisi yok ya da "
            "botun rolü bu rolden daha aşağıda!"
        )
    except Exception as error:
        await ctx.send(f"⚠️ Beklenmeyen bir hata oluştu: {str(error)}")


@bot.command(name="cmd")
async def show_commands_list(ctx):
    embed = discord.Embed(
        title="🌿 İklim Botu Komut Menüsü",
        color=discord.Color.blue()
    )
    embed.add_field(
        name="📸 !check",
        value="Çevre fotoğrafı yükleyip yapay zekaya analiz ettirir.",
        inline=False
    )
    embed.add_field(
        name="📊 !puan",
        value="Kendi güncel puanınızı gösterir.",
        inline=True
    )
    embed.add_field(
        name="🏆 !puanlar",
        value="En çevreci üyeleri listeler.",
        inline=True
    )
    embed.add_field(
        name="🛒 !market",
        value="Mağazayı açar.",
        inline=True
    )
    embed.add_field(
        name="🛍️ !satınal [ad]",
        value="Belirtilen rütbeyi satın alır.",
        inline=True
    )
    embed.add_field(
        name="🛠️ !puanver [@üye] [miktar]",
        value="*(Yetkili)* Belirtilen üyeye puan ekler.",
        inline=False
    )
    embed.add_field(
        name="⚙️ !puanset [@üye] [puan]",
        value="*(Yetkili)* Belirtilen üyenin puanını ayarlar.",
        inline=False
    )
    await ctx.send(embed=embed)


# --- YETKİLİ KOMUTLARI ---
@bot.command(name="puanver")
@has_authority_role()
async def give_points_to_member(
    ctx,
    target_member: discord.Member = None,
    amount: int = None
):
    if target_member is None or amount is None:
        await ctx.send(
            "⚠️ Yanlış kullanım! Doğru format: `!puanver @kullanıcı [miktar]`"
        )
        return

    if amount <= 0:
        await ctx.send("⚠️ Verilecek puan miktarı 0'dan büyük olmalıdır!")
        return

    try:
        await add_user_score(target_member.id, amount)
        updated_score = await get_user_score(target_member.id)
        await ctx.send(
            f"✅ Başarılı! {ctx.author.mention} isimli yetkili, "
            f"{target_member.mention} kullanıcısına **{amount} Puan** verdi. "
            f"Yeni Puanı: **⭐ {updated_score}**"
        )
    except Exception as error:
        await ctx.send(f"⚠️ Puan verilirken bir hata oluştu: {str(error)}")


@bot.command(name="puanset")
@has_authority_role()
async def set_points_of_member(
    ctx,
    target_member: discord.Member = None,
    target_score: int = None
):
    if target_member is None or target_score is None:
        await ctx.send(
            "⚠️ Yanlış kullanım! Doğru format: `!puanset @kullanıcı [puan]`"
        )
        return

    if target_score < 0:
        await ctx.send("⚠️ Ayarlanacak puan miktarı 0'dan küçük olamaz!")
        return

    try:
        await set_user_score(target_member.id, target_score)
        await ctx.send(
            f"✅ Başarılı! {ctx.author.mention} isimli yetkili, "
            f"{target_member.mention} kullanıcısının puanını direkt "
            f"**⭐ {target_score}** olarak ayarladı."
        )
    except Exception as error:
        await ctx.send(f"⚠️ Puan ayarlanırken bir hata oluştu: {str(error)}")


# --- BOT EVENTLERİ ---
@bot.event
async def on_ready():
    print(f'🤖 {bot.user.name} aktif!')
    try:
        create_table_query = """
            CREATE TABLE IF NOT EXISTS kullanicilar (
                user_id TEXT PRIMARY KEY,
                uint_puan INTEGER DEFAULT 0
            )
        """
        await execute_database_query(create_table_query)
        print("✓ Veritabanı başarıyla bağlandı.")
    except Exception as error:
        print(f"❌ Veritabanı oluşturma hatası: {str(error)}")
    load_machine_learning_model()


if __name__ == "__main__":
    bot.run(BOT_TOKEN)
