# 🧠 Makhos ML Training Pipeline

Machine Learning pipeline สำหรับ train neural network เล่นหมากฮอส (Thai Checkers)

## 🚀 เริ่มต้นที่นี่!

**👉 อ่าน `COLAB_QUICKSTART.md` ก่อน** - มีคำสั่ง copy-paste พร้อมใช้ทุก cell

## 📋 ภาพรวม 3 Steps

1. **Generate Data** (2-3 ชม.) - AI vs AI self-play → 5,000 games → `training_data.npz`
2. **Train Model** (30-60 นาที) - Neural network training → `best_model.pt`
3. **Export** - Download model มาใช้งาน

**เวลารวม**: ~3-4 ชั่วโมงบน Google Colab (มี GPU ฟรี!)

## ⚡ สำคัญ: TypeScript vs Pure Python

- **TypeScript** (แนะนำ): 2-3 วิ/เกมส์ → 5000 games = **2-3 ชม.** ✅
- **Pure Python**: 20 วิ/เกมส์ → 5000 games = **27+ ชม.** ❌

→ **ใช้ TypeScript version เท่านั้น!**

---

## 📁 ไฟล์ในโฟลเดอร์นี้

```
ml/
├── COLAB_QUICKSTART.md   # 📘 คู่มือหลัก - อ่านไฟล์นี้!
├── README.md             # ไฟล์นี้ - สรุปภาพรวม
├── gen_data.py           # Step 1: Gen data (call TypeScript)
├── model.py              # Neural networks (SimpleMakhosNet, MakhosNet)
├── train.py              # Step 2-3: Train & save
└── requirements.txt      # Dependencies (torch, numpy)
```

---

## 📖 วิธีใช้

อ่านใน **`COLAB_QUICKSTART.md`** มีทุกอย่างครบ:
- Setup (1 นาที)
- Clone repo & install (1 นาที)
- Gen data (2-3 ชม.)
- Train model (30-60 นาที)
- Test & download

**อย่าลืมเปลี่ยน `YOUR_USERNAME` ตอน clone!**

---

## 📊 ข้อมูลเทคนิค

### Neural Network

**Input**: (6 planes, 32 squares)
- P1 men, P1 kings, P2 men, P2 kings, side to move, halfmove clock

**Output**:
- Policy: (32×32) move probabilities
- Value: [-1, 1] position evaluation

### Dataset (5,000 games @ 1000ms/move)
- ~200,000 positions
- 1.2-1.5 GB compressed
- 2-3 ชม. gen time บน Colab

### Model Architectures

**SimpleMakhosNet** (แนะนำ):
- MLP 3 layers × 512 units
- ~1.5M parameters
- Train เร็ว (30 นาที)

**MakhosNet** (advanced):
- ResNet, 6 blocks × 128 channels
- ~3M parameters
- Train ช้ากว่า (1 ชม.)

---

## ⚙️ Options

### เวลาต่อตา
| ms/move | คุณภาพ | ใช้เมื่อ |
|---------|--------|---------|
| 500 | พอใช้ | ทดสอบ |
| 1000 | ดี | **แนะนำ** |
| 1200 | ดีมาก | เหมือน manual AI |

### Gen data
```bash
# ทดสอบ (100 games, 5 นาที)
python gen_data.py --total_games 100 --batch_size 100 --time_per_move 500

# แนะนำ (5000 games, 2-3 ชม.)
python gen_data.py --total_games 5000 --batch_size 1000 --time_per_move 1000

# Production (10k games, 5-6 ชม.)
python gen_data.py --total_games 10000 --batch_size 2000 --time_per_move 1000
```

### Resume หาก disconnect
```bash
# รันคำสั่งเดิมอีกรอบ แล้วเลือก 'y'
python gen_data.py --total_games 5000 --batch_size 1000 --time_per_move 1000
```

---

## 💡 Tips

1. **เริ่มจากเล็ก**: ทดสอบ 100 games ก่อน
2. **เปิด GPU**: Runtime → Change runtime type → GPU (เร็วขึ้น 5-10x)
3. **ดู overfitting**: เช็ค val_loss vs train_loss
4. **Resume ได้**: Disconnect แล้วรันคำสั่งเดิมอีกรอบ

## 🐛 แก้ปัญหา

**"No games were generated"**
- เช็ค Node.js: `node --version`
- เช็ค tsx: `npx tsx --version`
- รัน `npm install` ที่ root

**"Out of memory" ตอน train**
- ลด `--batch_size` (ลอง 16 หรือ 32)
- ใช้ SimpleMakhosNet แทน MakhosNet

**Train ช้า**
- เปิด GPU ใน Colab
- ใช้ SimpleMakhosNet

---

## 🎯 Next Steps

พรุ่งนี้:
1. Push code ขึ้น GitHub
2. เปิด Colab
3. รัน COLAB_QUICKSTART.md ตามลำดับ
4. รอ 3-4 ชม. → ได้ trained model!

**สนุกกับการ train! 🚀**