# Accelerate_bitsandbytes_aya-expanse-8b_4bit



from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from accelerate import Accelerator
import torch

# إعداد Accelerator لإدارة الموارد (CPU و GPU)
accelerator = Accelerator()

# إعداد التحميل إلى 4 بت
quantization_config = BitsAndBytesConfig(load_in_4bit=True)  # تغيير إلى 4 بت

# تحميل النموذج والمحول باستخدام 4-bit quantization و device_map='auto'
model_id = 'CohereForAI/aya-expanse-8b'
model = AutoModelForCausalLM.from_pretrained(model_id, 
                                             device_map='auto',
                                             quantization_config=quantization_config)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# نقل النموذج والمحول إلى الجهاز المدعوم من Accelerator
model, tokenizer = accelerator.prepare(model, tokenizer)

# إعداد الإدخال للمطالبة "How are you?"
input_text = "Who is Napoleon Bonaparte?"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)

# الحصول على استجابة النموذج
with torch.no_grad():
    outputs = model.generate(input_ids, max_length=512)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

# عرض الاستجابة
print(f"Response: {response}")



Loading checkpoint shards: 100%
 4/4 [00:57<00:00, 12.66s/it]
Response: Who is Napoleon Bonaparte?
Napoleon Bonaparte was a French military leader and emperor who rose to prominence during the French Revolution and led France from 1799 to 1815. He is considered one of the greatest military commanders in history, and his legacy continues to shape European politics and culture.
Born on August 15, 1769, in Ajaccio, Corsica, Napoleon showed early signs of intelligence and ambition. He received a thorough education, excelling in mathematics and languages. After joining the French Army at 17, Napoleon quickly rose through the ranks due to his strategic brilliance and military prowess.
His rise to power began during the French Revolution, where he played a crucial role in suppressing royalist rebellions. In 1799, Napoleon staged a coup d'état, becoming the First Consul of France. He then embarked on a series of military campaigns, conquering much of Europe and establishing a vast empire.
As Emperor of the French, Napoleon implemented significant reforms, including the Napoleonic Code, which standardized laws and established civil equality. He also made contributions to education, administration, and the legal system. However, his empire faced challenges, and after a series of defeats, he was exiled to the island of Elba in 1814.
Napoleon escaped and returned to France in 1815, sparking the Hundred Days. His final defeat at the Battle of Waterloo in 1815 marked the end of his rule. Napoleon spent the rest of his life under British surveillance on the island of Saint Helena, where he died in 1821.
Napoleon's impact extends beyond his military conquests. His strategic thinking, political reforms, and legal contributions have left an indelible mark on modern Europe. His life and legacy continue to fascinate historians, scholars, and the general public alike.
Would you like to know more about Napoleon's military campaigns or his impact on European history?
