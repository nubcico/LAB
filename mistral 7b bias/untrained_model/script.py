# THIS SCRIPT IS FOR DATA CLEANING

with open("D:\\Damir\\Downloads\\gai_proj_res\\untrained_model\\gai_test_f1_res_untrained_exp2.txt", encoding='utf-8') as f:
    lines = f.readlines()
all_labels = set()
true_labels = []
temp = []
count3 = 0
# print(len(lines))
for line in lines:
    # for label in line.split('\t')[0].split('-'):
        # all_labels.update(label)
    if '\t' in line and len(line.split('\t')) > 1:
        count3 += 1
        # print(count3)
        each_list = []

        temp = line.split('\t')[0].split('-')
        for ind in temp:
            each_list.append(ind.strip())
            all_labels.add(ind.strip())
        true_labels.append(each_list)
    # print(line.split('\t')[0].split('-'))
    # if len(line.split('\t')) > 1:
    #     predict_labels = line.split('\t')[1]
    #     print(i, predict_labels)
# print(all_labels)
f.close()
# print(len(true_labels))
with open("D:\\Damir\\Downloads\\gai_proj_res\\untrained_model\\gai_test_f1_res_untrained_exp2.txt", encoding='utf-8') as f:
    new_lines = f.readlines()
# print(len(new_lines))
if '' in all_labels:
    all_labels.remove('')
# all_labels.remove("'Coming home meant giving back for Pensacola native DC Reeves Editor's note: This is the final installment of a four")
# all_labels.remove("'Former University of Michigan team doctor investigated for multiple sex abuse complaints Palm Springs, California — The University of Michigan is investigating several \"disturbing and very serious\" allegations of sexual abuse against a now'")
# print(all_labels)
count = 0
count2 = 0



# print(len(zip(new_lines, true_labels)))
with open("cleaned_data_untrained_exp2.txt", 'a', encoding='utf-8') as f2: 
    for line in new_lines:
        count += 1
        # print(count)
        
        if '\t' in line and len(line.split('\t')) > 1:
            # print(count2)
            pred_labels = ''
            # count = 0
            # print(line.split('\t')[1])
            # print(line.split('\t')[1])
            row_true_labels = ''
            for word in true_labels[count2]:
                row_true_labels += word.strip() + ' '
            # print(row_true_labels[:-2])
            for line_labels in [line.split('\t')[1]]:
                # print(line_labels)
                for label in line_labels.split(' '):
                    # print(label.strip())
                    # if label.strip() in all_labels:

                    pred_labels += label.strip() + ' '
            # print(pred_labels[:-2])

            if row_true_labels[:-1] == '':
                count2 += 1
                continue

            if pred_labels == '':
                row = row_true_labels[:-1] + '\t\n'

            else :
                row = row_true_labels[:-1] + '\t' + pred_labels[:-1] + '\n'
            f2.write(row)
            count2 += 1

# print(count3)
# print(count)
# print(count2)
