# -*- coding: utf-8 -*-
"""
Created on Thu May 20 20:36:26 2021

@author: AM4
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May 19 21:13:16 2021

@author: AM4
"""

"""BERT Fine-Tuning Sentence Classification

Код базируется на https://colab.research.google.com/drive/1Y4o3jh3ZH70tl6mCd76vz_IxX23biCPP

# BERT Fine-Tuning Tutorial with PyTorch

By Chris McCormick and Nick Ryan

В коде используется библиотека [transformers](https://github.com/huggingface/transformers) 

Пред использованием необходимо установить пакеты:

conda install -c conda-forge transformers

"""

import torch
import numpy as np
import os
import pandas as pd


# Проверяем доступна ли GPU и задаем вычислительное устройство
if torch.cuda.is_available():    
    device = torch.device("cuda")
    print('Available GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU')
    device = torch.device("cpu")


# Загрузка данных реализована на основе pandas dataframe
df = pd.read_csv("./data/dataset_3_classes.csv")###поставил свой датасет 

print('В наборе предложений: {:,}\n'.format(df.shape[0]))

# Пример
df.sample(10)

# Нас интересуют метки классов и сами предложения, на них мы будем обучать нашу сеть
sentences = df.sentence.values
labels = df.label.values

# Следующий этап - токенизация - разбиение предложений на слова
from transformers import BertTokenizer

# Используем BERT tokenizer, но созданный на основе словаря
tokenizer = BertTokenizer('./data/vocab_rutoxic.txt', do_lower_case=True, do_basic_tokenize=True, never_split=None)

##### размер нового словаря
tokenizer.vocab_size

# максимальный размер предложения существенно вырос
sl = [len(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sen))) for sen in sentences]
print('Максимальная длина предложения: ', max([len(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sen))) for sen in sentences]))

# посмотрим сколько предложений имеет длину более 64 символа
value_c = pd.Series(sl).value_counts()
print('Предложений длиннее 64 токена: ', sum(value_c[64:]))

# будем их обрезать

input_ids = np.zeros((len(sentences),64))

# Каждое предложение энкодится по отдельности
for s,i in zip(sentences,range(len(sentences))):
    enc_s = tokenizer.encode(s,                      
                        add_special_tokens = True,
                        padding = 'max_length',
                        max_length = 64,
                        truncation = True
                   )
    input_ids[i,]=enc_s


# Создаем attention mask для виртуальных токенов
attention_masks = []

for s in input_ids:
    att_mask = [int(id_ > 0) for id_ in s]
    attention_masks.append(att_mask)


# Формируем тестовый и валидационный набор
from sklearn.model_selection import train_test_split

train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels, test_size=0.1)
train_masks, validation_masks, _, _ = train_test_split(attention_masks, labels, test_size=0.1)

# все конвертируем в тензоры
train_inputs = torch.tensor(train_inputs)
validation_inputs = torch.tensor(validation_inputs)

train_labels = torch.tensor(train_labels)
validation_labels = torch.tensor(validation_labels)

train_masks = torch.tensor(train_masks)
validation_masks = torch.tensor(validation_masks)

# теперь можно создавать Dataset и DataLoader
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

batch_size = 4

train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)


# теперь можно переходить к заданию модели

from transformers import BertForSequenceClassification, BertConfig

# Загрузка теперь делается через конфигурационный файл, в котором изменен размер словаря
configuration = BertConfig.from_pretrained('./data/config_rutoxic.json', num_labels=3)###Добавил 3тий класс
model = BertForSequenceClassification(configuration)

# Отправляем модель на GPU
if torch.cuda.is_available():
    model.cuda()

# Задаем оптимизатор
from torch.optim import AdamW

optimizer = AdamW(model.parameters(),
                  lr = 2e-5,
                  eps = 1e-8
                )

from transformers import get_linear_schedule_with_warmup

# Количество эпох обучения
epochs = 4

# Шагов обучения = number of batches * number of epochs.
total_steps = len(train_dataloader) * epochs

# scheduler - планировщик изменяющий скорость обучения
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)

# функция вычисления точности обучения
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


import time
import datetime
import random

# Задаем seed
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# Тут храним наши лоссы
loss_values = []

# Цикл обучения будет состоять из обучения и валидации
for epoch_i in range(0, epochs):
    
    ################ Часть обучения #####################
    
    print("")
    print('Эпоха {:} из {:} '.format(epoch_i + 1, epochs))
    
    t0 = time.time()

    total_loss = 0
    
    model.train()

    for step, batch in enumerate(train_dataloader):

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        
        model.zero_grad()

        outputs = model(b_input_ids.to(torch.long), 
                    token_type_ids=None, 
                    attention_mask=b_input_mask.to(torch.long), 
                    labels=b_labels.to(torch.long))
       
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        scheduler.step()
        
        if step % 10 == 0 and not step == 0:
            time_elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - t0))
            print(' Батч {:>4,} из {:>4,}. Затраченное время: {:}. Ошибка: {:}.'.format(step, len(train_dataloader), time_elapsed, loss))


    avg_train_loss = total_loss / len(train_dataloader)            
    
    loss_values.append(avg_train_loss)

    print("")
    print(" Средний loss: {0:.2f}".format(avg_train_loss))
    print(" Обучение эпохи прошло за: {:}".format(time.strftime("%H:%M:%S", time.gmtime(time.time() - t0))))
        
    ################ Часть валидации #####################

    print("\n Validation...")
    t0 = time.time()

    model.eval()

    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    for batch in validation_dataloader:
        
        batch = tuple(t.to(device) for t in batch)
        
        b_input_ids, b_input_mask, b_labels = batch
        
        with torch.no_grad():        
            outputs = model(b_input_ids.to(torch.long), 
                    token_type_ids=None, 
                    attention_mask=b_input_mask.to(torch.long), 
                    labels=b_labels.to(torch.long))
        
        logits = outputs.logits

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        
        eval_accuracy += tmp_eval_accuracy

        nb_eval_steps += 1

    print("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))
    print("  Валидация прошла за: {:}".format(time.strftime("%H:%M:%S", time.gmtime(time.time() - t0))))



# Сохраняем обученную модель
output_dir = './model_save_mydata/'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)


# Можно построить график обучения 
import matplotlib.pyplot as plt

plt.plot(loss_values, 'b-o')
plt.title("Training loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()


# Загрузим предобученную модель
model_dir = './model_save_mydata/'
model = BertForSequenceClassification.from_pretrained(model_dir)
tokenizer = BertTokenizer.from_pretrained(model_dir)

model.to(device)


# Проверим как оно работает
sentence = 'Учитель забыл выключить микрофон и запел.'###Изменил пример 
enc_s = tokenizer.encode(sentence,                      
                        add_special_tokens = True,
                        padding = 'max_length',
                        max_length = 64,
                        truncation = True
                   )

input_ids = np.array(enc_s)

attention_mask = [int(id_ > 0) for id_ in input_ids]

model.eval()
batch = tuple(t.to(device) for t in torch.Tensor([input_ids, attention_mask]))
b_input_ids, b_input_mask = batch
with torch.no_grad():
    outputs = model( b_input_ids.unsqueeze(0).to(torch.long), token_type_ids=None, attention_mask=b_input_mask.unsqueeze(0))
    
logits = outputs.logits
logits = logits.detach().cpu().numpy()
predicted_label = np.argmax(logits, axis=1).flatten()
print(predicted_label)
# Финальная точность модели в процентах
final_accuracy = eval_accuracy / nb_eval_steps
print("\nТочность модели:", round(final_accuracy * 100, 2), "%")


###добавил вывод слуайных примеров
label_names = {
    0: "новости",
    1: "технологии",
    2: "юмор"
}

examples = df.sample(5)

for i, row in examples.iterrows():
    print("\nТекст:", row["sentence"])
    print("Класс:", row["label"], f"({label_names[int(row['label'])]})")
