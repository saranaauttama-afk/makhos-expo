# Watcher Guide

คู่มือสั้น ๆ สำหรับรัน `newAz` local watcher ผ่าน `Command Prompt (cmd)`

ใช้กับโฟลเดอร์นี้:

- repo: `d:\My App\makhos-v2`
- drive dir: `G:\My Drive\makhos_az_v5`

---

## 1. เช็กสถานะ pipeline

ใช้คำสั่งนี้ก่อน ถ้าอยากดูว่า training ไปถึงไหนแล้ว และมี request / result / decision ล่าสุดอะไรบ้าง

```cmd
cd /d "d:\My App\makhos-v2" && python newAz\check_status.py --drive-dir "G:\My Drive\makhos_az_v5"
```

ถ้าเห็น:

- `pending eval : none`
- `pipeline handshake is working`

แปลว่าระบบโดยรวมยังโอเค

---

## 2. เช็กว่า watcher ยังรันอยู่ไหม

ใช้คำสั่งนี้ใน `cmd`

```cmd
powershell -Command "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'eval_local.py' } | Select-Object ProcessId, CommandLine"
```

ความหมาย:

- ถ้ามี output ขึ้นมา = watcher ยังรันอยู่
- ถ้าไม่ขึ้นอะไรเลย = watcher ไม่ได้รันอยู่

---

## 3. ปิด watcher เก่า

ถ้าเพิ่งแก้ `eval_local.py` หรือไม่แน่ใจว่ามี watcher เก่าค้างอยู่ไหม ให้ปิดก่อนแล้วค่อยเปิดใหม่

```cmd
powershell -Command "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'eval_local.py' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"
```

ถ้าไม่มี process อยู่ คำสั่งนี้ก็จะเงียบ ๆ ไป ไม่ถือว่าเป็นปัญหา

---

## 4. เปิด watcher ใหม่

เปิด `cmd` แล้วรัน:

```cmd
cd /d "d:\My App\makhos-v2" && python newAz\eval_local.py --drive-dir "G:\My Drive\makhos_az_v5" --watch
```

สำคัญ:

- หน้าต่างนี้ต้องปล่อยค้างไว้
- ถ้าไม่มี error และ prompt ไม่กลับมา แปลว่ากำลัง watch อยู่
- พอมี request ใหม่ มันจะเริ่ม eval เอง

---

## 5. ลำดับที่แนะนำเวลาไม่แน่ใจ

ถ้าจำไม่ได้ว่าตอนนี้ watcher รันอยู่ไหม ใช้ลำดับนี้ได้เลย:

1. เช็กสถานะ pipeline
2. ปิด watcher เก่า
3. เปิด watcher ใหม่

คำสั่งชุดนี้ใช้งานได้ทันที:

```cmd
cd /d "d:\My App\makhos-v2" && python newAz\check_status.py --drive-dir "G:\My Drive\makhos_az_v5"
```

```cmd
powershell -Command "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'eval_local.py' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"
```

```cmd
cd /d "d:\My App\makhos-v2" && python newAz\eval_local.py --drive-dir "G:\My Drive\makhos_az_v5" --watch
```

---

## 6. หลังเปิด watcher แล้วควรเห็นอะไร

เมื่อมี checkpoint ใหม่ เช่น `iter_0079`

- Colab จะสร้าง request
- watcher จะพิมพ์ว่า `processing request iter_0079.request.json`
- จากนั้นจะเขียน
  - `external_eval/results/iter_0079.result.json`
  - `external_eval/decisions/iter_0079.decision.json`

ถ้า checkpoint เริ่มแข็งพอ รอบใหม่จะมี `practical verify` เพิ่มเข้ามาใน output ด้วย

---

## 7. หมายเหตุ

- ถ้า Colab หลุด แต่ไฟล์ request ถูกเขียนแล้ว local watcher ยัง eval ต่อได้
- ถ้าปิดเครื่อง local watcher ก็หยุด แต่ไฟล์ request ไม่หาย
- เปิดเครื่องใหม่แล้วรัน watcher ซ้ำ มันจะมาเก็บงานค้างต่อได้
