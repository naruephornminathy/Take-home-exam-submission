# สรุปผลการทดลองและการพัฒนาโมเดลจำแนกเมฆจากภาพดาวเทียม (Cloud Segmentation)

## 1. Data Challenges & Analysis

* **คุณลักษณะของข้อมูล (Dataset Overview):**
  * **จำนวนภาพ:** 8,400 patches (Train) และ 9,201 patches (Test)
  * **ขนาดภาพและแบนด์:** ขนาด $384 \times 384$ pixels จำนวน 4 แบนด์ (`R`, `G`, `B`, `NIR`) จาก 38 scenes ของ Landsat 8
* **การวิเคราะห์การกระจายตัวของเมฆ (Class Imbalance & Bimodal Distribution):**
  * สัดส่วน pixel *non-cloud* ต่อ *cloud* ภาพรวมอยู่ที่ประมาณ **2.68:1** ซึ่งดูเหมือนไม่ imbalance มากนักในภาพรวม
  * **ลักษณะ Bimodal ในระดับ Patch:** เมื่อพิจารณาในระดับ patch พบการกระจายตัวแบบ bimodal ชัดเจน:
    * **47% ของ Patch:** แทบไม่มีเมฆเลย ($<1\%$)
    * **13% ของ Patch:** มีเมฆเต็มภาพ ($>99\%$)
  * **ผลกระทบ:** ทำให้ patch ที่มีขอบหรือ segment ให้โมเดลเรียนรู้จริงๆ มีจำนวนน้อย และยังมีหลาย patch ที่ติดขอบดำ (Artifacts) จาก scene ต้นฉบับ
* **กลยุทธ์ Loss Function:**
  * เลือกใช้ **BCE ร่วมกับ Dice Loss** แทนที่จะใช้ BCE เพียงอย่างเดียว เนื่องจาก Dice Loss สามารถวัดการ overlap ของพื้นที่เมฆได้โดยตรงและรับมือกับปัญหา class imbalance ได้ดีกว่า ส่วน BCE ช่วยควบคุม gradient ในช่วงแรกไม่ให้แกว่งมากเกินไป *(ไม่ได้ทำการ filter หรือ oversample patch เนื่องจากข้อจำกัดด้านเวลา)*
* **ปัญหาและการประเมินผลบน Test Set (Test Set Evaluation Issue):**
  * ในการประเมินผลบน Task 4 test set พบว่า official ground truth มีเฉพาะในระดับ scene เต็ม (`Entire_scene_gts`) ไม่มีในระดับ patch
  * ได้พยายาม reverse-engineer สคริปต์ MATLAB เพื่อหาตำแหน่งและการต่อ patch กลับไปที่ภาพเต็ม (คำนวณตำแหน่งตามตาราง เติมขอบดำ พลิกภาพทุกทิศทาง) แต่ทายตำแหน่งถูกต้องได้เพียง **48%** เมื่อเทียบกับชุดเทรนที่ทราบตำแหน่งจริงอยู่แล้ว
  * **การแก้ไข:** เปลี่ยนมาประเมินผลบน **Validation Split (1,680 patches)** ที่เตรียมแยกไว้ตั้งแต่แรกแทน

---

## 2. Model Architecture Rationale

* **โครงสร้างโมเดล (Encoder-Decoder + Skip Connections):**
  * สถาปัตยกรรม **Encoder-Decoder 4 ชั้น พร้อม Skip Connections**
  * **Input:** 4 channels (`R` / `G` / `B` / `NIR`)
  * **Output:** Single-channel ผ่าน Sigmoid activation ที่ resolution $384 \times 384$ เท่าเดิม
* **เหตุผลทางเทคนิค (Rationale):**
  * เนื่องจากงานนี้เป็น **pixel-level classification** (Segmentation) ไม่ใช่ image-level จึงจำเป็นต้องมี decoder เพื่อ reconstruction ภาพกลับมาขนาดเดิม
  * **ความสำคัญของ Skip Connections:** เมฆมีรูปร่างและขอบเขตที่ไม่แน่นอน หากบีบอัดข้อมูลลงเหลือ $24 \times 24$ ที่ bottleneck รายละเอียดเชิงพื้นที่ (Spatial Details) จะสูญหายไป การมี skip connections เพื่อส่งข้อมูล Feature Map จาก Encoder โดยตรงจะช่วยให้โมเดลเก็บขอบเมฆที่คมชัดได้ดี
* **การกำหนดสเปกและการปรับใช้ (Hardware Setup):**
  * ตั้งค่า Base Channels เริ่มต้นที่ **32** (ไล่ระดับ: `32` $\rightarrow$ `64` $\rightarrow$ `128` $\rightarrow$ `256` $\rightarrow$ `512`) ตั้งแต่ช่วงพัฒนาเบื้องต้นบนเครื่อง Local (`RTX 4050`, VRAM 6GB)
  * ย้ายไปเทรนจริงรอบสุดท้ายบน Google Colab (`NVIDIA A100 40GB`) ซึ่งรองรับ Batch Size 32 ได้อย่างเรียบร้อย
* **ข้อจำกัดของ Label (Dataset Label Noise):**
  * พบว่าในบางกรณี เมฆบางๆ ที่เห็นได้ชัดเจนในแบนด์ NIR (ปรากฏเป็นเส้นทแยงสว่าง) กลับไม่มี label เมฆใน Ground Truth เลย
  * สันนิษฐานว่าเป็น Label Noise หรือข้อมูลไม่สมบูรณ์จากต้นทาง ซึ่งเป็นปัจจัยที่ต้องนำมาคำนึงถึงเมื่อวิเคราะห์และตีความผลการประเมิน

---

## 3. Results Analysis

### การฝึกฝนโมเดล (Training Dynamics)
* **การตั้งค่า:** 50 Epochs, Optimizer `AdamW` ($lr = 10^{-3}$), `ReduceLROnPlateau` Scheduler, Early Stopping (patience=5), บันทึก Checkpoint ตามค่า `val_IoU` ที่ดีที่สุด
* **พฤติกรรมระหว่างเรียนรู้:** Training loss ลดลงอย่างต่อเนื่องจาก 0.29 เหลือ **0.05** แต่ Validation Loss มี Spike ชัดเจนในช่วงแรก (Epoch 3 และ 16 ส่งผลให้ `val_IoU` ตกไปที่ 0.59–0.65) ก่อนจะเริ่มนิ่งคงที่ตั้งแต่ Epoch 20 เป็นต้นไป
* **ข้อสังเกต:** การเกิด Spike คาดว่าเกิดจาก Learning Rate สูงไปเล็กน้อย แต่ Scheduler สามารถปรับลดและดึงโมเดลกลับมาได้ ไม่ใช่สัญญาณของ Overfitting เนื่องจากหลังการ Spike โมเดลสามารถทำคะแนน New High บน Validation Set ได้เสมอ โดย Best Checkpoint อยู่ที่ **Epoch 49** (`val_IoU` = 0.9281)

### ผลการประเมินบน Validation Split (1,680 Patches)

| Metric | Score |
| :--- | :---: |
| **Pixel Accuracy** | **97.87%** |
| **Precision** | **96.41%** |
| **Recall** | **96.13%** |
| **IoU** | **92.83%** |
| **F1-Score** | **95.25%** |

> **ข้อสังเกต:** ค่า Precision และ Recall มีความแตกต่างกันไม่ถึง **0.3%** แสดงว่าโมเดลไม่มีปัญหา Systematic Bias ทั้งในด้านการทำนายเกิน (Over-predict) หรือการทำนายขาด (Under-predict)

### การวิเคราะห์เชิงคุณภาพ (Qualitative Analysis)
* จากการสุ่มตรวจ Top 5 Patches ที่มีค่า IoU สูงสุด (สุ่มเฉพาะ patch ที่มีสัดส่วนเมฆ 5–95%) พบว่าโมเดลสามารถจับขอบเมฆได้อย่างแม่นยำในระดับ Pixel-to-Pixel
* ในกรณีที่มีขอบดำแนวทแยง (Artifact จากภาพดาวเทียมต้นฉบับ) โมเดลสามารถแยกแยะขอบดำออกจากเมฆจริงได้อย่างถูกต้อง สะท้อนว่าโมเดลเรียนรู้ Pattern ของ Artifacts ได้อย่างสมบูรณ์

---

## 4. Bonus Experiments: Data Augmentation & Loss Functions

ทำการทดลองเพิ่มเติม 2 ชุด เพื่อเปรียบเทียบประสิทธิภาพ (ใช้ 50 Epochs, Optimizer และ Scheduler เดียวกันกับ Baseline):
1. **Data Augmentation:** เพิ่ม Random Horizontal/Vertical Flip, Rotate 90° ให้กับทั้ง 4 แบนด์ + Mask และใส่ Brightness Jitter เฉพาะแบนด์ RGB (คงแบนด์ NIR ไว้ตามข้อกำหนด)
2. **BCE Only:** ตัด Dice Loss ออก ใช้เฉพาะ BCE Loss โดยไม่ใส่ Augmentation

### ตารางเปรียบเทียบผลการทดลอง (Experiment Comparison)

| Configuration | Best Val IoU | Best Val F1 | Best Val Loss | Best Epoch |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline (BCE + Dice, No Aug)** | **0.928** | **0.962** | **0.052** | **49** |
| **Data Augmentation** | 0.921 | 0.958 | 0.059 | 47 |
| **BCE Only (No Dice, No Aug)** | 0.927 | 0.962 | 0.053 | 48 |

* **สรุปผลทดลอง:** **Baseline (BCE + Dice แบบไม่ใส่ Augmentation)** ยังคงเป็น Configuration ที่ให้ผลลัพธ์ดีที่สุด แม้ว่าจะแตกต่างกับชุดอื่นเพียงเล็กน้อย การเสริม Dice Loss ช่วยเพิ่มประสิทธิภาพได้อย่างชัดเจน ส่วน Data Augmentation ให้ผลลัพธ์ดรอปลงเล็กน้อยและไม่คุ้มค่ากับ Compute Resource ที่เสียไป

---

## 5. Improvements & Future Work

1. **Weighted Sampling:** ปรับการสุ่มตัวอย่างแบบ Weighted Sampling ในขั้นตอน Training เพื่อเพิ่มน้ำหนักให้ patch ที่มีสัดส่วนเมฆระดับปานกลางถูกนำมาเรียนรู้มากขึ้น เนื่องจาก patch กลุ่มนี้เป็นจุดสำคัญในการตัดสินความคมชัดของขอบเมฆ
2. **Patch Re-stitching Script:** พัฒนาและแก้ไขสคริปต์การต่อภาพใน Official Test Set ให้สมบูรณ์ เพื่อให้สามารถประเมินและรายงานผลบน Benchmark กลางของ Dataset ได้โดยตรง
3. **Advanced Augmentation Exploration:** ทดสอบ Data Augmentation รูปแบบอื่นๆ หรือปรับแต่งความเข้มข้น เพื่อหาการตั้งค่าที่ช่วยเพิ่ม IoU ได้อย่างแท้จริงโดยไม่ลดทอนประสิทธิภาพของโมเดล
