# 🧠 Makhos ML Training Pipeline

Machine Learning pipeline สำหรับ train neural network เล่นหมากฮอส (Thai Checkers)

## 🚀 เริ่มต้นที่นี่!

**👉 อ่าน `colab_quickstart.md` ก่อน** - มีคำสั่ง copy-paste พร้อมใช้ทุก cell

## 📋 ภาพรวม 3 Steps

1. **Generate Data** (~11 ชม.) - AI vs AI self-play → 500 games → `training_data_500_t1000.npz`
2. **Train Model** (30-60 นาที) - Neural network training → `best_model.pt`
3. **Test & Export** - ทดสอบและ download model มาใช้งาน

**เวลารวม**: ~12 ชั่วโมงบน Google Colab (รันข้ามคืน)

## ⚡ ความเร็วจริง

- **TypeScript** (time=1000ms): ~50-80 วินาที/เกม
  - 100 เกม = **1.4 ชั่วโมง** (ทดสอบ)
  - 500 เกม = **11 ชั่วโมง** (แนะนำ) ✅
  - 5,000 เกม = **110 ชั่วโมง** (production)

→ **เริ่มจาก 500 เกมก่อน เพื่อทดสอบ model!**

---

## 📁 ไฟล์ในโฟลเดอร์นี้

```
ml/
├── colab_quickstart.md   # 📘 คู่มือหลัก - อ่านไฟล์นี้!
├── README.md             # ไฟล์นี้ - สรุปภาพรวม
├── gen_data.py           # Step 1: Gen data (call TypeScript via nvm)
├── model.py              # Neural networks (SimpleMakhosNet, MakhosNet)
├── train.py              # Step 2-3: Train & save
└── requirements.txt      # Dependencies (torch, numpy)
```

---

## 📖 วิธีใช้

อ่านใน **`colab_quickstart.md`** มีทุกอย่างครบ:
- **Cell 1**: Setup Drive + PyTorch + Node.js v20 via nvm (3-5 นาที)
- **Cell 2**: Clone repo & npm install (1 นาที)
- **Cell 3-4**: Gen data (1.4-11 ชม.)
- **Cell 5**: Inspect data
- **Cell 6**: Train model (30-60 นาที)
- **Cell 7**: Test model
- **Cell 8**: Download (optional)

**อย่าลืมเปลี่ยน `YOUR_USERNAME` ตอน clone!**

---

## 📊 ข้อมูลเทคนิค

### Neural Network

**Input**: (6 planes, 32 squares)
- P1 men, P1 kings, P2 men, P2 kings, side to move, halfmove clock

**Output**:
- Policy: (32×32) move probabilities
- Value: [-1, 1] position evaluation

### Dataset ขนาดต่างๆ

| เกม | Time/move | Positions | Size | Gen time |
|-----|-----------|-----------|------|----------|
| 100 | 500ms | ~5,000 | ~100 MB | 1.4 ชม. |
| 500 | 1000ms | ~25,000 | ~500 MB | 11 ชม. |
| 5,000 | 1000ms | ~250,000 | ~5 GB | 110 ชม. |

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

### Gen data (ใน Colab)

```python
# ทดสอบ (100 games, 1.4 ชม.)
generate_data(
    total_games=100,
    batch_size=100,
    time_per_move=500,
    output="/content/drive/MyDrive/makhos_ml/training_data_100.npz"
)

# แนะนำสำหรับครั้งแรก (500 games, 11 ชม.)
generate_data(
    total_games=500,
    batch_size=50,
    time_per_move=1000,
    output="/content/drive/MyDrive/makhos_ml/training_data_500_t1000.npz",
    batch_dir="/content/drive/MyDrive/makhos_ml/batches_500"  # เก็บ batch ใน Drive
)

# Production (5000 games, 110+ ชม. - แบ่งทำหลายวัน)
generate_data(
    total_games=5000,
    batch_size=100,
    time_per_move=1000,
    output="/content/drive/MyDrive/makhos_ml/training_data_5000.npz",
    batch_dir="/content/drive/MyDrive/makhos_ml/batches_5000"
)
```

### Resume หาก Colab disconnect

```python
# รันคำสั่งเดิมอีกรอบ
# โค้ดจะเจอ batch ที่เซฟไว้ใน Drive แล้วถาม:
# "Resume from checkpoint? (y/n)"
# ตอบ 'y' → เล่นต่อจาก batch ถัดไป!
```

---

## 💡 Tips

1. **เริ่มจาก 500 เกม**: เพื่อทดสอบ model ก่อน (11 ชม.)
2. **เก็บ batch ใน Drive**: เพื่อ resume ได้ถ้า disconnect
3. **เปิด GPU สำหรับ training**: Runtime → Change runtime type → GPU (เร็วขึ้น 5-10x)
4. **ดู overfitting**: เช็ค val_loss vs train_loss
5. **รันข้ามคืน**: เปิดเครื่อง + browser ทิ้งไว้

## 🐛 แก้ปัญหา

**"Cannot find module 'node:path'"**
- Node.js เวอร์ชันเก่า → ใช้ nvm ติดตั้ง v20 (ดูใน Cell 1)
- เช็ค: `!bash -c "source ~/.nvm/nvm.sh && node --version"` ควรได้ v20.x.x

**"No games were generated"**
- เช็คว่า npm install เสร็จหรือยัง
- Reload module: `del sys.modules['gen_data']`
- ลอง restart runtime

**"Out of memory" ตอน train**
- ลด `batch_size=32` หรือ `16`
- ใช้ SimpleMakhosNet แทน MakhosNet (ResNet)

**Train ช้า**
- เปิด GPU ใน Colab (Runtime → GPU)
- ใช้ SimpleMakhosNet (~30 นาที แทน 1 ชม.)

---

## 🎯 Status & Next Steps

### 📅 วันนี้ (เสร็จแล้ว):
- ✅ Setup Colab pipeline (Node.js v20 via nvm)
- ✅ ทดสอบ gen data 100 เกม (1.4 ชม.)
- ✅ แก้ปัญหา Node.js compatibility
- ✅ แสดง real-time progress

### 🌙 คืนนี้ (กำลังรัน):
- ⏳ **Gen 500 games (time=1000ms)**
- ⏳ รอ ~11 ชั่วโมง (ตอนนี้เกม ~70/500 แล้ว)
- 📁 เซฟ batch ใน Drive ทุก 50 เกม

### 🌅 พรุ่งนี้ (To-do):
1. ✅ เช็คว่า 500 เกมเสร็จหรือยัง
2. 📊 Inspect data (Cell 5)
3. 🧠 Train model 30 epochs (~30-40 นาที)
4. 🧪 Test model (Cell 7)
5. 📈 ประเมินผล → ตัดสินใจว่าจะเพิ่มข้อมูลหรือไม่
6. 💾 Download model (ถ้าดี)

### 🚀 แผนระยะยาว:
- ถ้า model 500 เกมดี → พอใช้ได้แล้ว
- ถ้าต้องการ model แกร่งกว่า → เพิ่มเป็น 1,000-5,000 เกม (แบ่งทำหลายวัน)

**สนุกกับการ train! 🎮🤖**