1. Data Challenges
Dataset 8,400 patches(train) 9,201 patches(test) ขนาด 384×384, 4 แบนด์ (R,G,B,NIR) จาก 38 scenes ของ Landsat 8 
• จากการวิเคราะห์ข้อมูล พบว่าสัดส่วน pixel non-cloud ต่อ cloud ทั้งชุดอยู่ที่ประมาณ 2.68:1 ซึ่งดูเหมือนไม่ imbalance มากนัก แต่เมื่อดูในระดับ patch จะเห็นว่าการกระจายเป็นแบบ bimodal ชัดเจน โดย 47% ของ patch แทบไม่มีเมฆเลย (<1%) และ 13% มีเมฆเต็มภาพ (>99%) ส่งผลให้ patch ที่มี segment ให้โมเดลเรียนรู้จริงๆ มีจำนวนน้อย และยังมีหลาย patch ที่ติดขอบดำจาก scene ต้นฉบับด้วย
• เลือกใช้ loss function แบบ BCE ร่วมกับ Dice แทนที่จะใช้ BCE อย่างเดียว เพราะ Dice สามารถวัดการ overlap ได้โดยตรงและรับมือกับปัญหา imbalance ได้ดีกว่า ส่วน BCE ช่วยควบคุม gradient ในช่วงแรกไม่ให้แกว่งมากเกินไป (ไม่ได้ทำการ filter หรือ oversample patch เนื่องจากข้อจำกัดด้านเวลา)
• ในการประเมินผลบน Task 4 test set พบว่า official ground truth มีเฉพาะในระดับ scene เต็ม (Entire_scene_gts) ไม่มีในระดับ patch จึงต้อง reverse-engineer สคริปต์ MATLAB เพื่อหาตำแหน่งและการต่อ patch กลับไปที่ภาพเต็ม โดยลองหลายวิธี เช่น คำนวณตำแหน่งตามตาราง เติมขอบดำ และพลิกภาพในทุกทิศทาง แต่สามารถทายตำแหน่งถูกต้องได้เพียง 48% เมื่อเทียบกับชุดเทรนที่รู้ตำแหน่งจริงอยู่แล้ว สุดท้ายจึงเปลี่ยนมาประเมินผลบน validation split (1,680 patches) ที่เตรียมไว้ตั้งแต่แรกแทน
2. Architecture Rationale
Encoder-Decoder 4 ชั้น + Skip Connections รับ input 4 channels (R/G/B/NIR) output single-channel ผ่าน Sigmoid ที่ resolution 384×384 เท่าเดิม
• เหตุผลที่เลือกสถาปัตยกรรมนี้เพราะงานนี้เป็น pixel-level classification ไม่ใช่ image-level จึงต้องมี decoder เพื่อสร้างภาพกลับมาขนาดเดิม ที่สำคัญคือการใช้ skip connections เนื่องจากเมฆมีรูปร่างและขอบเขตที่ไม่แน่นอน หากบีบข้อมูลลงเหลือ 24×24 ที่ bottleneck รายละเอียดเชิงพื้นที่จะหายไป ถ้าไม่มี skip connections ส่งข้อมูลจาก encoder มาช่วย โมเดลจะไม่สามารถเก็บขอบเมฆที่คมชัดได้
• ตั้งค่า base channel ที่ 32 (ไล่ 32-64-128-256-512) ตั้งแต่ช่วงพัฒนาเบื้องต้นบนเครื่อง local (RTX 4050, VRAM 6GB) ก่อนจะย้ายไปเทรนจริงรอบสุดท้ายบน Colab (A100 40GB) ซึ่งสามารถรองรับ batch size 32 ได้
• พบข้อจำกัดของ label คือในบางกรณีเมฆบางที่เห็นชัดในแบนด์ NIR (เส้นทแยงสว่าง) กลับไม่มี label เมฆใน ground truth เลย ซึ่งน่าจะเกิดจาก label ที่ไม่สมบูรณ์หรือมี noise เป็นข้อจำกัดของ dataset ที่ต้องคำนึงถึงเมื่อตีความผลประเมิน
3. Results Analysis
• ผลการเทรน
50 epochs, AdamW (lr=1e-3), ReduceLROnPlateau, early stopping patience=5, save checkpoint ตาม val_IoU ดีที่สุด
Training loss ลดจาก 0.29 เหลือ 0.05 แต่ validation loss มี spike ชัดเจนในช่วงแรก (epoch 3, 16 val_IoU ตกเหลือ 0.59–0.65) ก่อนจะนิ่งตั้งแต่ epoch 20 เป็นต้นไป สันนิษฐานว่า learning rate สูงไปเล็กน้อยแต่ scheduler สามารถปรับกลับได้ ไม่ใช่สัญญาณของ overfitting เพราะหลัง spike ทุกครั้ง โมเดลสามารถทำคะแนน val score สูงสุดใหม่ได้ โดย checkpoint ที่ดีที่สุดอยู่ที่ epoch 49 (val_IoU 0.9281)
• ผลประเมินบน Validation split:
Metric Score
Pixel Accuracy 97.87%
Precision 96.41%
Recall 96.13%
IoU  92.83%  
F1 95.25%
Precision และ Recall มีค่าห่างกันไม่ถึง 0.3% แสดงว่าโมเดลไม่มี bias แบบ systematic ทั้งในกรณี over-predict หรือ under-predict
• จากการตรวจสอบเชิงคุณภาพโดยสุ่ม Top 5 patches ที่มี IoU สูงสุด (เฉพาะ patch ที่มีเมฆ 5–95%) พบว่าโมเดลสามารถจับขอบเมฆได้แม่นยำในระดับ pixel-to-pixel และในบางกรณีที่มีขอบดำแนวทแยง โมเดลก็ยังแยกขอบดำออกจากเมฆจริงได้อย่างถูกต้อง ยืนยันว่าโมเดลสามารถแยก artifact ของภาพดาวเทียมออกจากเมฆได้จริง
4. Improvements & Future Work
• ปรับการสุ่มตัวอย่างแบบ weighted sampling ในขั้นตอนการเทรน เพื่อให้ patch ที่มีสัดส่วนเมฆระดับปานกลางถูกนำมาเรียนรู้มากขึ้น เพราะ patch กลุ่มนี้เป็นตัวตัดสินความคมของขอบเมฆจริง
• พัฒนา script สำหรับการต่อภาพใน official test set ให้สมบูรณ์ เพื่อจะสามารถรายงานผลบน benchmark กลางของ dataset ได้โดยตรง โดยไม่ต้องใช้ validation set แทน
• เพิ่มขั้นตอน Data Augmentation (เช่น Flip, Rotation, RGB Brightness Jitter) เพื่อทดสอบว่าจะช่วยเพิ่มค่า IoU ได้มากน้อยแค่ไหน โดยในงานนี้แยกการทดลอง augmentation ออกมาเพื่อเปรียบเทียบกับ baseline
Bonus: Data Augmentation กับ Loss Function Experiments
ทำการรันการทดลองเพิ่มอีก 2 ชุด (50 epochs, optimizer และ scheduler เดียวกัน) 
1.  Data Augmentation ใส่ Random Flip (H/V), Rotate 90° กับทั้ง 4 แบนด์ + Mask และใส่ Brightness Jitter เฉพาะแบนด์ RGB (เว้น NIR ไว้ตามข้อกำหนดโจทย์)
2.  BCE อย่างเดียว (ตัด Dice ออก) โดยไม่ใส่ Augmentation เพื่อดูผลของตัว Loss
Configuration Best Val IoU Best Val F1 Best Val Loss Best Epoch
Baseline (BCE+Dice, No Aug) 0.928 0.962 0.052 49
+ Data Augmentation 0.921 0.958 0.059 47
BCE Only (No Dice, No Aug) 0.927 0.962 0.053 48
สรุปว่า Baseline (BCE+Dice, ไม่ทำ augmentation) ยังคงเป็น configuration ที่ดีที่สุดในทั้ง 3 ชุด แม้ผลจะต่างจากอีก 2 ชุดไม่มาก การเพิ่ม Dice loss ให้ประโยชน์เล็กน้อยแต่ชัดเจน ส่วน augmentation ไม่คุ้มกับค่าใช้จ่ายด้าน compute ที่เพิ่มขึ้น
