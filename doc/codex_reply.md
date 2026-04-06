# Codex Reply

เอกสารนี้สรุปว่า Codex เห็นปัญหาอะไรในชุดเทรน AlphaZero เดิม, ทำไมต้องแก้, และในรอบล่าสุดได้แก้อะไรไว้บ้างในชุดไฟล์ `colab/codex/`

## เป้าหมาย

โจทย์จริงไม่ใช่แค่ให้โมเดล `loss` ลงหรือชนะ checkpoint เก่า แต่คือ

- ไม่ให้ checkpoint ช่วงหลัง ๆ เกิด regression แบบเงียบ ๆ
- วัดความแข็งแรงให้ใกล้คู่ต่อสู้จริงมากขึ้น
- เพิ่มโอกาสให้ checkpoint ที่เลือกไป deploy ชนะ `mm11` ได้จริง

## ปัญหาหลักที่พบในแนวทางเดิม

### 1. Objective ที่ใช้คัด checkpoint ยังไม่ตรงเป้า

ลูปเดิมวัดหลัก ๆ ด้วย

- `curr_net` vs `best_net`
- `curr_net` vs random
- `curr_net` vs minimax ตื้น

แต่เป้าหมายจริงคือชนะ `mm11`

ผลคืออาจได้ checkpoint ที่ดูดีใน arena ภายใน แต่ไม่ใช่ตัวที่เก่งสุดกับ minimax ลึกจริง

### 2. Self-play drift ง่าย

เมื่อ `curr_net` เล่นกับตัวเองเป็นหลัก distribution ของเกมใหม่จะวิ่งตาม policy ล่าสุดเร็วมาก ถ้า policy ล่าสุดเริ่มเอนเอียงหรือเรียนรู้สิ่งที่ exploit ตัวเองได้ โมเดลจะดูดีใน training loop แต่แย่ลงเมื่อเจอคู่ต่อสู้ต่างสไตล์

### 3. Evaluation เดิมยังบอกไม่พอว่าโมเดลเก่ง “ทั้งสองฝั่ง” หรือไม่

เกมนี้มี side bias ชัด ถ้าวัดแค่ค่าเฉลี่ยรวม มีโอกาสโปรโมตโมเดลที่เก่งเฉพาะฝั่งเดียว

### 4. Training loop ยังไม่มีระบบกัน regression ที่ชัดเจน

ถ้าโมเดลเริ่มถอยลงจริง ลูปเดิมยังไม่มีทั้ง

- การลด LR ตาม metric เป้าหมาย
- การ rollback กลับ checkpoint ที่ดีกว่า
- การดึง hard positions จากเกมแพ้กลับมาสอนใหม่

## แนวคิดของชุดแก้ล่าสุด

รอบล่าสุดใน `colab/codex/train_az.py` ไม่ได้แก้แค่ tuning เล็ก ๆ แต่เปลี่ยนแนวคิดของ loop ให้ “optimize ตาม practical strength” มากขึ้น โดยมี 4 แกนหลัก

1. `Opponent Pool`
2. `Opening Suite`
3. `Plateau LR + Rollback`
4. `Loss Mining`

## สิ่งที่แก้ใน `colab/codex/train_az.py`

### 1. เปลี่ยน self-play จาก `curr vs curr` อย่างเดียว เป็น actor-based training

เพิ่มโครงสร้าง actor ให้เกมหนึ่งเกมสามารถเกิดจากคู่ต่อสู้หลายแบบได้ เช่น

- network vs network
- network vs minimax
- network vs historical checkpoint

เหตุผล:

- ลด overfit กับ policy ล่าสุด
- ทำให้ replay buffer มีความหลากหลายมากขึ้น
- ใกล้กับ practical training มากกว่า self-play แบบปิดโลก

### 2. เพิ่ม `Opponent Pool`

ลูป self-play ตอนนี้สุ่มคู่ต่อสู้จากหลายแหล่ง เช่น

- `best_net`
- `target_net`
- historical checkpoints
- minimax บาง depth
- `curr_net` เอง

และมีการ refresh pool ระหว่างเทรน

เหตุผล:

- ถ้าเจอแต่ตัวเองอย่างเดียว โมเดลจะ specialize ง่าย
- opponent pool บังคับให้โมเดลรับมือหลายสไตล์
- historical checkpoints ช่วย anchor ความเก่งย้อนหลัง

ผลที่คาดหวัง:

- checkpoint หลัง ๆ จะ regress ยากขึ้น
- ความเก่ง transfer ไป battle จริงดีขึ้น

### 3. เพิ่ม `Opening Suite` แบบ deterministic

มีการสร้างชุด opening positions คงที่ล่วงหน้า แล้วใช้ suite นี้วัดทุก checkpoint แบบเดิมทุกครั้ง

metric ที่เพิ่มเข้ามา:

- opening overall
- opening P1
- opening P2
- opening floor = ค่าต่ำสุดระหว่าง P1/P2

เหตุผล:

- ช่วยเห็นว่าความเก่งจริงสม่ำเสมอหรือไม่
- กันกรณีโมเดลเก่งเฉพาะฝั่งเดียว
- ใช้เป็น tie-break หรือ guard rail ตอนเลือก `target_best`

ผลที่คาดหวัง:

- checkpoint ที่ถูกเลือกจะ balanced กว่า
- ลดโอกาส deploy โมเดลที่ side-broken

### 4. เปลี่ยน scheduler เป็น `ReduceLROnPlateau`

เดิมแนวคิดเป็น scheduler ตามเวลา

รอบนี้เปลี่ยนให้ลด LR ตาม metric เป้าหมายจริง คือ `wr_mm11`

เหตุผล:

- ถ้า metric จริงไม่ดีขึ้น การลด LR มีความหมายกว่าการหมุนตาม iteration อย่างเดียว
- ช่วยให้ training ปรับตัวตามคุณภาพจริง ไม่ใช่ตามเวลาอย่างเดียว

ผลที่คาดหวัง:

- late-stage drift ลดลง
- checkpoint ช่วงท้ายเสถียรมากขึ้น

### 5. เพิ่ม `Rollback`

ถ้า `wr_mm11` และ opening metrics ตกต่ำกว่าตัว `target_best` ต่อเนื่องตาม threshold ที่กำหนด ระบบจะ

- rollback `curr_net` กลับไป `target_best`
- reset optimizer/scheduler ใหม่
- ลด LR ลงอีกขั้น

เหตุผล:

- กันการไหลลงเหวแบบต่อเนื่อง
- ถ้าหลุดจาก good regime แล้ว ระบบควรกลับไปฐานที่ดีแทนที่จะฝืนไปต่อ

ผลที่คาดหวัง:

- ใช้ compute คุ้มขึ้น
- ไม่ต้องคอยมานั่งเฝ้าว่า run นี้เสียไปแล้วหรือยัง

### 6. เพิ่ม `Loss Mining`

หลัง checkpoint จะมีการเอา `curr_net` ไปเล่นกับ minimax เป้าหมาย ถ้าแพ้ จะดึงตำแหน่งช่วงท้ายเกมที่แพ้กลับมา แล้วใช้ move จาก minimax เป็น policy target เพื่อใส่กลับ replay buffer

เหตุผล:

- self-play อย่างเดียวอาจไม่พอสำหรับ weakness บาง pattern
- loss mining บังคับให้โมเดลเรียนจาก hard failures ตรง ๆ

ผลที่คาดหวัง:

- practical strength ต่อ `mm11` เพิ่มไวขึ้น
- แก้จุดอ่อนเฉพาะทางได้ดีกว่า pure self-play

### 7. เพิ่ม target selection ที่เข้มขึ้น

เดิม `target_best` ดูแค่ win rate หลัก ๆ

ตอนนี้การอัปเดต `target_best` ดูร่วมกันอย่างน้อย 2 มิติ

- `wr_mm11`
- opening quality โดยเฉพาะ side floor

เหตุผล:

- ไม่อยากได้ checkpoint ที่ชนะรวมดี แต่ฝั่งหนึ่งพัง
- target ที่ดีควรเก่งทั้งตาม objective หลักและมีเสถียรภาพเชิงโครงสร้าง

### 8. เพิ่ม state ที่จำเป็นต่อการ resume จริง

ตอน save state ตอนนี้มีเพิ่ม เช่น

- `target_best_wr`
- `target_best_suite`
- `target_best_side_floor`
- `rollback_streak`

เหตุผล:

- ถ้า Colab หลุดแล้วกลับมา ระบบต้องต่อ logic เดิมได้
- ไม่งั้น plateau/rollback behavior จะขาดตอน

### 9. ปรับ progress table ให้ใช้ metric ที่มีความหมายกว่าเดิม

ตอนดู progress ตอนนี้จะเห็นเพิ่ม เช่น

- `vs_mm11`
- opening score
- side floor
- current LR
- จำนวน samples ที่มาจาก loss mining

เหตุผล:

- ช่วยให้ตัดสินใจเร็วขึ้นว่ารันนี้กำลังดีขึ้นจริงหรือไม่
- ไม่ต้องเดาจาก `p_loss` / `v_loss` อย่างเดียว

### 10. แก้ export ONNX ให้โหลด architecture จาก checkpoint metadata

จุดนี้ยังคงอยู่จากรอบก่อน

เหตุผล:

- กันการ export checkpoint ด้วยสถาปัตยกรรมผิดตัว
- สำคัญเมื่อมีหลายรุ่นของโมเดลในระบบ

## สิ่งที่แก้ใน `colab/codex/makhos_engine.py`

ยังคงแนวทางเดิมจากรอบก่อน คือทำให้ heuristic minimax ฝั่ง Python ใกล้กับฝั่ง TypeScript มากขึ้น โดยเพิ่มส่วนอย่างเช่น

- material score
- piece-square tables
- mobility
- back-rank guard
- simplification bonus

เหตุผล:

- minimax ที่ใช้วัดใน Colab ควรใกล้กับ minimax ที่ใช้วัดจริงในแอป
- ไม่อย่างนั้นจะเกิดปัญหา “ผ่านใน Colab แต่ตกใน battle จริง”

## ทำไมต้องแยกไว้ใน `colab/codex/`

ผู้ใช้ขอให้ไฟล์เดิมยังคงอยู่ ดังนั้นแนวทางคือ

1. ไม่แก้ทับไฟล์ใน `colab/`
2. ย้ายชุดที่ Codex ปรับทั้งหมดไปอยู่ใน `colab/codex/`
3. ให้ทดลองเทียบกันได้แบบปลอดภัย

ข้อดี:

- revert ง่าย
- review ง่าย
- เปรียบเทียบกับ baseline ได้ชัด
- ไม่กระทบ workflow เดิมทันที

## วิธีใช้งาน

ถ้าต้องการลองแนวทาง Codex ให้ใช้ไฟล์ในโฟลเดอร์นี้แทน

- `colab/codex/train_az.py`
- `colab/codex/makhos_engine.py`
- `colab/codex/network_az.py`
- `colab/codex/mcts_az.py`

แล้วอัปโหลดไปยัง Colab/Drive เหมือน flow เดิม

## สิ่งที่ควรคาดหวัง

### ข้อดี

- การคัด checkpoint จะตรงกับเป้าหมาย `mm11` มากขึ้น
- regression ตอน checkpoint หลัง ๆ น่าจะลดลง
- โมเดลที่ได้มีโอกาส balanced ระหว่าง P1/P2 มากขึ้น
- training loop มีระบบกู้ตัวเองเมื่อเริ่มถอย

### ต้นทุนที่เพิ่มขึ้น

- checkpoint eval จะช้าลง
- logic ซับซ้อนขึ้น
- log/state มีหลายตัวแปรมากขึ้น

## ข้อควรระวัง

### 1. ยังไม่การันตีว่ารอบเดียวจะชนะ `mm11`

ชุดแก้นี้ทำให้ objective ตรงขึ้นและ practical กว่าเดิม แต่ไม่ได้แปลว่าทุกรันจะชนะ `mm11` ทันที

### 2. loss mining เป็นแนวทางเชิง practical มากกว่า pure AlphaZero

ถ้าเป้าหมายคือ pure research purity อาจไม่ใช่ทางที่สะอาดที่สุด
แต่ถ้าเป้าหมายคือ “เอาชนะ `mm11`” มันคุ้มมาก

### 3. Security issue ยังอยู่

ในไฟล์ยังมี `EMAIL_PASSWORD` แบบ plaintext ควรย้ายไปใช้ environment variable ภายหลัง

## ข้อเสนอเชิงปฏิบัติ

ถ้าจะใช้ชุดนี้จริง แนะนำลำดับนี้

1. เริ่มจาก checkpoint ที่ดีที่สุดตอนนี้
2. ใช้ `target_best.pt` เป็น candidate หลักสำหรับ export/deploy
3. ดู `wr_vs_minimax11`, opening suite, และ side floor เป็น metric หลัก
4. battle ซ้ำฝั่ง TypeScript เพื่อยืนยัน practical strength ก่อนสรุปผล

## บทสรุป

รอบแรกของ Codex เน้นแก้ให้ objective ตรงกับ `mm11`

รอบล่าสุดนี้ขยายต่อไปอีกขั้น โดยเปลี่ยนจาก “ปรับค่าบางตัว” เป็น “เปลี่ยน training loop ให้กัน regression และเรียนจากจุดแพ้จริง”

แกนหลักของรอบล่าสุดคือ

1. `Opponent Pool`
2. `Opening Suite`
3. `Plateau LR + Rollback`
4. `Loss Mining`

ทั้งหมดนี้ถูกใส่ไว้ใน `colab/codex/` เพื่อให้ทดลองได้เต็มที่โดยไม่แตะไฟล์ต้นฉบับ
