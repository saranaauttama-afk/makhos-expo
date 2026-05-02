# AZ Training Handoff - 2026-04-25

ไฟล์นี้คือสรุปสำหรับเปิด chat ใหม่เพื่อทำงานต่อเรื่องเทรน AlphaZero / AZ ของ Makhos โดยไม่หลุดไปฝั่ง UI

## เป้าหมายหลัก

ต้องการเทรนโมเดล AZ ให้เล่น Makhos ได้แข็งแรงขึ้น เป้าระยะยาวคือชนะหรืออย่างน้อยสู้ minimax ลึกสูง ๆ เช่น mm9/mm11 ได้

ตอนนี้ข้อสรุปสำคัญคือ:

- โมเดลล้วนยังไม่พร้อมใช้เป็น AI หลักในเกมมือถือ
- โมเดลมีอาการเดินดัน ๆ ขึ้นไปเอง และบางจังหวะเดินให้กินแบบไม่มีเหตุผล
- self-play ต่อจาก checkpoint เดิมมีแนวโน้ม regression / catastrophic forgetting
- ควรหยุดการลอง self-play ยาว ๆ แบบเดิมก่อน เพราะเปลือง Colab และผลไม่นิ่ง
- ทางที่ควรไปต่อคือ teacher / supervised training จาก minimax tactical data แล้วค่อยกลับไป self-play

## สถานะ baseline ที่น่าเก็บ

checkpoint ที่ยังน่าสนใจที่สุดตอนนี้คือ `iter_0079`

ผลที่จำได้จากการ eval:

- vs random: 100%
- vs mm3: 100%
- vs mm5: 100%
- vs mm7: ประมาณ 50%
- vs mm9: 75%
- vs mm11: เริ่มแพ้ และแต่ละเกมช้ามาก ผู้ใช้หยุดก่อนจบ

หมายเหตุ:

- `iter_0079` ไม่ได้ชนะ mm11 แต่ดูเป็นจุดที่ดีที่สุดก่อนเกิด regression
- checkpoint หลังจากนี้ เช่น 89, 99, 109, 119, 129, 139 มีหลายรอบที่ดูเหมือนดีขึ้นบางด้าน แต่ external eval มักไม่ดีขึ้นจริงหรือถอย

## ปัญหาที่เจอจาก self-play เดิม

อาการหลัก:

- training loss ลง แต่ความสามารถจริงไม่ได้ดีขึ้นเสมอ
- บางรอบ p_loss ลง แต่ v_loss ขยับขึ้น
- ช่วง 79 -> 89 บาง config ดูพอไปได้ แต่ 89 -> 99 -> 109 มักเริ่มถอย
- quick eval ใน Colab มี sample น้อย จึงหลอกได้ง่าย
- external eval กับ mm7/mm9/mm11 สำคัญกว่า quick eval
- mm11 eval ใช้เวลานานมาก บางครั้ง 20 ชั่วโมงขึ้นไป

ข้อสรุป:

- อย่าตัดสินจาก p_loss/v_loss อย่างเดียว
- อย่าปล่อยเทรนยาวถ้ายังไม่ผ่าน tactical / mm7 / mm9 ที่สั้นกว่า
- อย่าเอา mm11 เป็น gate แรก เพราะแพงเกินไป

## Paths และ command สำคัญ

Repo local:

```cmd
d:\My App\makhos-v2
```

Google Drive local:

```cmd
G:\My Drive\makhos_az_v_teacher
```

Colab path:

```text
/content/drive/MyDrive/makhos_az_v_teacher
```

local eval checkpoint:

```cmd
cd /d "d:\My App\makhos-v2"
python newAz\eval_local.py --drive-dir "G:\My Drive\makhos_az_v_teacher" --checkpoint iter_0079
```

local watcher:

```cmd
cd /d "d:\My App\makhos-v2"
python newAz\eval_local.py --drive-dir "G:\My Drive\makhos_az_v_teacher" --watch
```

status:

```cmd
cd /d "d:\My App\makhos-v2"
python newAz\check_status.py --drive-dir "G:\My Drive\makhos_az_v_teacher"
```

## Eval pipeline ที่แก้ไปแล้ว

`newAz/eval_local.py` เคยมีปัญหา stale result:

- checkpoint ชื่อเดิมถูกเขียนทับ เช่น `iter_0099.pt`
- result เก่า `iter_0099.result.json` ยังอยู่
- watcher อาจเข้าใจว่าประเมินแล้ว ทั้งที่ checkpoint เป็นคนละรอบ

สิ่งที่แก้ไปแล้ว:

- eval เขียน progress เป็นช่วง ๆ ระหว่าง eval
- watcher เช็ค mtime ของ checkpoint กับ result
- ถ้า checkpoint ใหม่กว่า result จะ reevaluate
- stale eval เก่าถูกย้ายไป backup

backup path:

```text
G:\My Drive\makhos_az_v_teacher\external_eval\backup\stale_eval_20260424_140246
```

ไฟล์ที่ย้าย:

- `iter_0099.result.json`
- `iter_0099.decision.json`
- `iter_0109.result.json`
- `iter_0109.decision.json`

## Notebook/config ล่าสุดที่เคยใช้

ไฟล์หลัก:

- `newAz/train_colab.ipynb`
- `G:\My Drive\makhos_az_v_teacher\train_colab.ipynb`

ค่าที่เคยตั้งเพื่อกลับไปจาก baseline:

```python
FORCE_BASELINE_ITER = 79
FORCE_RESET_TRAIN_STATE = True
FORCE_CLEAR_REPLAY_BUFFER = True
FORCE_RESET_LOG_CURSOR = True

LR = 8e-5
N_SELFPLAY = 75
N_SIMS = 400
TRAIN_STEPS = 300
REPLAY_SIZE = 200_000
SELFPLAY_MINIMAX_DEPTHS = [5, 7]

ENABLE_LOSS_MINING = False
AUTO_LOSS_MINING_FROM_DECISIONS = False

STABILITY_REG_WEIGHT = 0.07
STABILITY_VALUE_WEIGHT = 0.35
RUN_BLOCK_ITERS = 20
```

ผลรอบล่าสุดที่ไม่ค่อยดี:

```text
iter 89/200
p_loss: 1.2367
v_loss: 0.2986
quick vs best: 25%
quick vs random: 100%
quick vs mm3: 100%
quick vs mm5: 100%
loss mining: off @d7 g3 t6 m36
LR: 8.00e-05
external eval queued for iter 0089
```

## แผนที่ควรทำต่อ: Teacher Training

หยุดการ self-play แบบเดิม แล้วสร้างรอบเทรนแบบ teacher-guided

แนวคิด:

- ใช้ `iter_0079` เป็นฐาน
- สร้าง dataset จาก minimax depth 7/9
- ให้ minimax เป็นครูบอก best move และ value
- เทรน supervised เพื่อแก้ tactical blunder ก่อน
- ค่อยกลับไป self-play หลังโมเดลไม่เดินพลาดง่าย ๆ

เหตุผล:

- ตอนนี้โมเดลลืม pattern ดี ๆ จาก checkpoint ก่อนหน้า
- self-play จากตัวเองที่ยังพลาด จะยิ่งผลิต data พลาดซ้ำ
- teacher data ช่วยบังคับให้ policy ไม่เดินให้กินง่าย ๆ
- ประหยัดกว่าเทรนยาวแล้วรอ mm11 หลายสิบชั่วโมง

## งานแรกของ chat ใหม่

ให้ทำ 3 อย่างนี้ก่อน อย่าเพิ่งแก้ UI

1. สร้าง `newAz/build_teacher_data.py`

หน้าที่:

- โหลด engine / rules / network ที่มีอยู่
- สร้างตำแหน่งจากหลายแหล่ง:
  - opening / random legal playout
  - minimax-guided playout
  - positions ที่มี forced capture
  - positions ที่ model กับ minimax เห็นไม่ตรงกัน
- label policy ด้วย minimax depth 7 หรือ 9
- label value ด้วย minimax score หรือ rollout outcome
- balance ฝั่ง P1/P2
- save เป็น `.npz` และ metadata `.json`

2. สร้าง `newAz/train_teacher.py`

หน้าที่:

- โหลด checkpoint `iter_0079.pt`
- train policy + value จาก teacher dataset
- มี anchor/regularization ไม่ให้ drift จาก baseline มากเกินไป
- save checkpoint ใหม่ เช่น `teacher_iter0079_d7_YYYYMMDD.pt`
- log loss แยก:
  - policy CE
  - value loss
  - anchor loss

3. สร้าง `newAz/TEACHER_TRAINING.md`

หน้าที่:

- อธิบาย concept
- คำสั่ง build dataset
- คำสั่ง train
- คำสั่ง eval
- วิธีเลือก checkpoint ที่ควรไปต่อ

## Eval policy ที่ควรใช้หลัง teacher training

อย่าเริ่มจาก mm11

ลำดับควรเป็น:

1. tactical fixed suite
2. vs random
3. vs mm3
4. vs mm5
5. vs mm7
6. vs mm9
7. mm11 เฉพาะ checkpoint ที่ผ่าน mm7/mm9 ชัดเจนแล้ว

กติกาตัดสินเบื้องต้น:

- ถ้าไม่ผ่าน mm5 ให้ทิ้ง
- ถ้า mm7 ต่ำกว่า 50% ยังไม่ควรไป mm9/mm11
- ถ้า mm9 ไม่อย่างน้อย 50% แบบนิ่ง อย่าเสียเวลา mm11
- ต้องดู P1/P2 แยก เพราะเคยมีอาการรุกดีแต่รับแย่ หรือกลับกัน

## สิ่งที่ต้องระวัง

- อย่าใช้ quick eval ใน Colab เป็นตัวตัดสินหลัก
- อย่าเปิด loss mining เร็วเกินไป
- อย่า train ต่อจาก checkpoint ที่ regression ชัด ถ้าไม่มีเหตุผล
- อย่า overwrite result แล้วเชื่อว่าคือผลเดิม
- อย่าเอา model-only ไปแทน minimax ในเกมจริงตอนนี้
- อย่าปล่อย mm11 ยาวถ้า mm7/mm9 ยังไม่นิ่ง

## สถานะ repo ตอนสร้าง handoff

งาน UI ล่าสุดมี uncommitted changes อยู่ 2 ไฟล์:

- `src/coreClaude/movegen.ts`
- `src/ui/HumanVsCodexArenaScreen.tsx`

เป็นงานเรื่อง multi-capture step animation/UI ไม่เกี่ยวกับ training

ถ้าเปิด chat ใหม่เพื่อเทรน ให้ ignore งาน UI ก่อน หรือ commit แยกภายหลัง

## Prompt สำหรับเปิด chat ใหม่

ใช้ข้อความนี้เปิด chat ใหม่ได้เลย:

```text
เราจะทำต่อเรื่อง AZ training ของ Makhos เท่านั้น ไม่ทำ UI ตอนนี้

อ่าน doc/az_training_handoff.md ก่อน

เป้าคือหยุด self-play เดิมที่ regression แล้วทำ teacher/supervised training จาก iter_0079:
- สร้าง newAz/build_teacher_data.py
- สร้าง newAz/train_teacher.py
- สร้าง newAz/TEACHER_TRAINING.md

ขอแบบใช้งานจริง ประหยัด Colab และมี eval gate ชัดเจนก่อนกลับไปเทรนยาว
```
