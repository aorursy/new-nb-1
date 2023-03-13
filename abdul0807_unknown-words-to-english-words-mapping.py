# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np

import re
from IPython.display import display
train = pd.read_csv("../input/train.csv", index_col=None)
print(train.shape)

display(train.head())

test = pd.read_csv("../input/test.csv", index_col=None)
print(test.shape)

display(test.head())
puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 
 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 
 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 
 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 
 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]
mispell_dict = {"aren't" : "are not",
"can't" : "cannot",
"couldn't" : "could not",
"didn't" : "did not",
"doesn't" : "does not",
"don't" : "do not",
"hadn't" : "had not",
"hasn't" : "has not",
"haven't" : "have not",
"he'd" : "he would",
"he'll" : "he will",
"he's" : "he is",
"i'd" : "I would",
"i'd" : "I had",
"i'll" : "I will",
"i'm" : "I am",
"isn't" : "is not",
"it's" : "it is",
"it'll":"it will",
"i've" : "I have",
"let's" : "let us",
"mightn't" : "might not",
"mustn't" : "must not",
"shan't" : "shall not",
"she'd" : "she would",
"she'll" : "she will",
"she's" : "she is",
"shouldn't" : "should not",
"that's" : "that is",
"there's" : "there is",
"they'd" : "they would",
"they'll" : "they will",
"they're" : "they are",
"they've" : "they have",
"we'd" : "we would",
"we're" : "we are",
"weren't" : "were not",
"we've" : "we have",
"what'll" : "what will",
"what're" : "what are",
"what's" : "what is",
"what've" : "what have",
"where's" : "where is",
"who'd" : "who would",
"who'll" : "who will",
"who're" : "who are",
"who's" : "who is",
"who've" : "who have",
"won't" : "will not",
"wouldn't" : "would not",
"you'd" : "you would",
"you'll" : "you will",
"you're" : "you are",
"you've" : "you have",
"'re": " are",
"wasn't": "was not",
"we'll":" will",
"didn't": "did not",
"tryin'":"trying"}
def clean_text(x):
    x = str(x)
    for punct in puncts:
        x = x.replace(punct, f' {punct} ')
    return x

def clean_numbers(x):
    x = str(x)
    x = re.sub(r'[0-9]{5,}', '#####', x)
    x = re.sub(r'[0-9]{4}', '####', x)
    x = re.sub(r'[0-9]{3}', '###', x)
    x = re.sub(r'[0-9]{2}', '##', x)
    
    return x

def _get_mispell(mispell_dict):
    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
    return mispell_dict, mispell_re

mispellings, mispellings_re = _get_mispell(mispell_dict)

def replace_typical_misspell(text):
    def replace(match):
        return mispellings[match.group(0)]
    
    text = mispellings_re.sub(replace, text)
    return text
train["question_text"] = train["question_text"].apply(lambda x: x.lower())
test["question_text"] = test["question_text"].apply(lambda x: x.lower())

train["question_text"] = train["question_text"].map(clean_text)
test["question_text"] = test["question_text"].map(clean_text)

train["question_text"] = train["question_text"].map(clean_numbers)
test["question_text"] = test["question_text"].map(clean_numbers)

train["question_text"] = train["question_text"].map(replace_typical_misspell)
test["question_text"] = test["question_text"].map(replace_typical_misspell)
def determine_unique_words(sent):
    unique_words = set()
    for records in sent:
        words = set([word for word in records.split()])
        unique_words.update(words)
    return unique_words
unique_words = determine_unique_words(list(train.question_text)+list(test.question_text))
print(len(unique_words))
unknown_words = []
for word in unique_words:
    if not re.match(r'[A-Za-z0-9#]', word) and len(word) > 1:
        unknown_words.append(word)
unknown_words = list(set(unknown_words))
len(unknown_words)
print(unknown_words[:50])
{'할': 'which', '猫头鹰': 'owl', 'पुरे': 'Enough', 'अष्टांगिक': 'Octagonal', 'आध्यात्मिकता': 'Spirituality', 'ബുദ്ധിമാനായ': 'Intelligent', 'बहार': 'spring', '陈': 'Chen', '她': 'she was', 'くわんぜおんぼさつ': 'Honey', '人民': 'people', 'صاحب': 'owner', 'मटका': 'pot', '阻天下悠悠之口': 'Block the mouth of the world', '在冰箱裡': 'in the refrigerator'}
if True == False:
    import time
    from tqdm import tqdm_notebook
    from googletrans import Translator

    word_to_english_map = {}
    for word in tqdm_notebook(unknown_words[0:500]):
        translator = None
        try:
            translator = Translator()
            english_word = translator.translate(word).text
            word_to_english_map[word] = english_word 
        except:
            print(f"can't find translation for word {word}")
            continue
words_to_english_map = {'隶书': 'Lishu', 'तपस्': 'Tapse', 'ให้': 'give', '宗教官': 'Religious officer', '没有神一样的对手': 'No god-like opponent', 'יהודיות': 'Jewish women', '我明明是中国人': 'I am obviously Chinese.', 'お早う': 'Good morning', 'मेरे': 'my', 'পরীক্ষা': 'Test', 'נעכטן': 'Yesterday', '中华民国': 'Republic of China', 'तकनीकी': 'Technical', 'यहाँ': 'here', '奇兵隊': 'Qibing', 'かﾟ': 'Or', '知识分子': 'Intellectuals', 'ごめなさい': "I'm sorry", '하기를': 'To', 'गाढवाचे': 'Donkey', 'इबादत': 'Worship', '千金': 'Thousand gold', 'ельзи': 'elzhi', '低端人口': 'Low-end population', '맛저': 'Mazur', 'दानि': 'Danias', 'юродство': 'foolishness', 'τον': 'him', '素质': 'Quality', '王晓菲': 'Wang Xiaofei', 'माहेर': 'Maher', 'لمصارى': 'For my advice', '送客！': 'see a visitor out!', '然而': 'however', 'जिग्यासा': 'Jigyaasa', '都市': 'city', 'ஜோடி': 'Pair', 'لاني': 'See', 'пельмени': 'dumplings', '请不要误会': "Please don't misunderstand", 'люис': 'Lewis', '不送！': 'Do not send!', 'के': 'Of', 'मस्का': 'Mascara', 'вoyѕ': 'voyce', 'बुन्ना': 'Bunna', '战略支援部队': 'Strategic support force', '平埔族': 'Pingpu', 'فهمت؟': 'I see?', 'कयामत': 'Apocalypse', 'ức': 'memory', 'ᗰeᑎ': 'ᗰe ᑎ', 'सेकना': 'Sexting', 'धकेलना': 'Shove', 'পারি': 'We can', 'مقام': 'Official', 'बेसति': 'Baseless', '歪果仁研究协会': 'Hazelnut Research Association', '短信': 'SMS', 'всегда': 'is always', '修身': 'Slim fit', 'إنَّ': 'that', '不是民国': 'Not the Republic of China', '写的好': 'Written well', 'õhtupoolik': 'afternoon', 'तालीन': 'Training', 'और': 'And', 'щит': 'shield', 'ᗩtteᑎtiᐯe': 'ᗩtte ᑎ ti ᐯ e', 'اسکوپیه': 'Script', 'çomar': 'fido', '和製英語': 'Japanglish', '吊着命': 'Hanging', 'много': 'much', 'समुराय': 'Samurai', 'भटकना': 'Wander', 'مقاومة': 'resistance', '싱관없어': 'I have no one.', '修身養性': 'Self-cultivation', 'मटका': 'pot', 'θιοψʼ': 'thiós', 'ㅎㅎㅎㅎ': 'Hehe', 'تساوى': 'Equal', 'बाट': 'From', '不地道啊。': 'Not authentic.', 'контакт': 'contact', '이런것도': 'This', 'तै': 'The', 'मेल': 'similarity', 'álvarez': 'Alvarez', 'नुक्स': 'Damage', '口訣': 'Mouth', 'масло': 'butter', 'परम्परा': 'Tradition', '学会': 'learn', 'کردن': 'Make up', 'öd': 'bile', 'टशन': 'Tashan', 'つらやましす': 'I am cheating', 'чего': 'what', '为什么总有人能在shoplex里免费获取iphone等重金商品？': 'Why do people always get free access to iphone and other heavy commodities in shoplex?', 'श्री': 'Mr', 'प्रेषक': 'Sender', 'خواندن': 'Read', 'बदलेंगे': 'Will change', 'बीएड': 'Breed', 'अदा': 'Paid', 'फैलाना': 'spread', 'ബുദ്ധിമാനായ': 'Intelligent', '谢谢六佬': 'Thank you, Liu Wei', 'উপস্থাপন': 'Present', 'بكلها': 'All of them', 'जाए': 'Go', '无可挑剔': 'Impeccable', 'आना': 'come', '太阴': 'lunar', 'القط': 'The cat', '있네': 'It is.', 'कर्म': 'Karma', 'आड़': 'Shroud', 'įit': 'IIT', 'जशने': 'Jashnay', 'से': 'From', 'åkesson': 'Åkesson', '乳名': 'Milk name', '我看起来像韩国人吗': 'Do I look like a Korean?', 'тнan': 'tn', 'العظام': 'Bones', '暹罗': 'Siam', '小肥羊鍋店': 'Little fat sheep pot shop', 'করেন': 'Do it', 'हरामी': 'bastard', 'डाले': 'Cast', 'استراحة': 'Break', '射道': 'Shooting', 'επιστήμη': 'science', 'χράομαι': "I'm glad", '脚踏车': 'bicycle', 'हिलना': 'To move', 'đa': 'multi', '標點符號': 'Punctuation', 'मैं': 'I', '不能化生水谷精微': "Can't transform the water valley", 'निवाला': 'Boar', 'दर्शनाभिलाषी': 'Spectacular', 'বাছাই': 'Picking', 'どくぜんてき': 'Dokedari', 'पीवें।': 'Pieve.', 'ʻoumuamua': 'Priority', 'में': 'In', 'चलाऊ': 'Run', 'निपटाना': 'To settle', 'ごはん': 'rice', 'चार्ज': 'Charge', 'शिथिल': 'Loosely', 'ястребиная': 'yastrebina', 'ложись': 'lie down', '李银河': 'Li Yinhe', 'へも': 'Also', 'हुलिया': 'Hulia', 'ऊब': 'Bored', 'छाछ': 'Buttermilk', 'عن': 'About', 'नहीं': 'No', 'जीव': 'creatures', 'जेहाद': 'Jehad', 'νερό': 'water', '열여섯': 'sixteen', 'मार': 'Kill', '早就': 'Long ago', '《齐天大圣》for': '"Qi Tian Da Sheng" for', 'الجنس': 'Sex', 'مع': 'With', 'रंगरलियाँ': 'Color palettes', 'जोल': 'Jol', 'बुद्धी': 'Intelligence', '罗思成': 'Luo Sicheng', '独善的': 'Self-righteousness', 'धर्मः': 'Religion', '中国大陆的同胞们': 'Chinese compatriots', '愛してない': 'I do not love you', '日本語が上手ですね': 'Japanese is good', 'ओठ': 'Lips', '如果你希望表達你的觀點': 'If you want to express your opinion', 'देशवाशी': 'Countryman', 'καί': 'and', '阮铭': 'Yu Ming', '跑步跑': 'Running', 'हो': 'Ho', '「褒められる」': '"Praised"', 'నా': 'My', '\x10new': '?new', 'ゆいがどくそん': 'A graduate student', '白语': 'White language', 'वह': 'She', '白い巨塔': 'White big tower', '로리이고': 'Lori.', 'परि': 'Circumcision', 'æj': 'Aw', 'انضيع': 'Lost', '平民苗字必称義務令': 'Civilian Miaozi must be called an obligation', 'ਸਿੰਘ': 'Singh', 'уже': 'already', 'đầu': 'head', '雨热同期': 'Rain and heat', 'घुलना': 'Dissolve', 'мис': 'Miss', 'て下さいませんか': 'Could it be?', '猫头鹰': 'owl', 'चढ़': 'Climbing', '漢音': 'Hanyin', 'यही': 'This only', '只有猪一样的队友': 'Only pig-like teammates', 'übersoldaten': 'about soldiers', 'αθ': 'Ath', 'ουδεις': 'no', '党国合一': 'Party and country', 'रेड': 'Red', 'ढोलना': 'Drift', 'शाक': 'Shake', '같이': 'together', '攻克': 'capture', 'özalan': 'exclusive', 'काम': 'work', 'चाहना': 'Wish', '坚持一个中国原则': 'Adhere to the one-China principle', '배우고': 'Learn', '柴棍': 'Firewood stick', 'उसे': 'To him', 'на': 'on', '無手': 'No hands', 'čvrsnica': 'Čvrsnica', 'ज़ार': 'Tsar', 'ˆo': 'O', '後宮甄嬛傳': 'Harem of the harem', '意音文字': 'Sound character', 'बहारा': 'Bahara', 'てみる': 'Try', '老铁': 'Old iron', '野比のび太': 'Nobita Nobita', 'याद': 'remember', 'پیشی': 'Surpass', 'توقعك': 'Your expectation', 'はたち': 'Slender', 'فزت': 'I won', '伊藤': 'Ito', 'कलेजे': 'Liver', 'αγεν': 'ruin', 'ìmprovement': 'Improvement', 'では': 'Then.', 'பார்ப்பது': 'Viewing', '只留清气满乾坤': 'Only stay clear and full of energy', '이정도쯤': 'About this', 'すてん': 'Sponge', 'ما': 'What', '海南人の日本': "Hainan people's Japan", '小鹿乱撞': 'very excited', 'ῥωμαῖοι': 'ῥomaşii', 'वारी': 'Vary', 'भोज': 'Feast', '陈庚': 'Chen Geng', 'कट्टरवादि': 'Fanatic', '凌霸': 'Lingba', 'पार': 'The cross', 'कुचलना': 'To crush', 'रहा': 'Stayed', 'हम': 'We', 'бойся': 'be afraid', '大刀': 'Large sword', 'ন্ন': 'Done', '汽车': 'car', 'কে': 'Who', 'الکعبه': 'Alkaebe', '网络安全法': 'Cybersecurity law', 'नेपाली': 'Nepali', 'পদের': 'Position', '\x92t': '??t', 'रास': 'Ras', 'لكل': 'for every', 'शूद्राणां': 'Shudran', '此问题只是针对外国友人的一次统计': 'This question is only a statistic for foreign friends.', 'ឃើញឯកសារអំពីប្រវត្តិស្ត្រនៃប្រាសាទអង្គរវត្ត': 'See the history of Angkor Wat', 'спасибо': 'thank', 'رب': 'God', 'वजूद': 'Non existent', 'पकड़': 'Hold', 'बरहा': 'Baraha', '雾草': 'Fog grass', 'мощный': 'powerful', 'বৃদ্ধাশ্রম': 'Old age', 'आई': 'Mother', 'खड़ी': 'Steep', '\x1aaùõ\x8d': '? aùõ ??', 'āto': 'standard', '在冰箱裡': 'in the refrigerator', 'đời': 'life', 'लड़का': 'The boy', 'τὸ': 'the', 'مدرسان': 'Instructors', 'பறை': 'Drum', 'índia': 'India', 'दिया': 'Gave', '小篆': 'Xiao Yan', '唐樂': 'Tang Le', '米国': 'USA', '文言文': 'Classical Chinese', 'उवाच': 'Uwach', 'هدف': 'Target', 'घेरा': 'Cordon', 'झूम': 'Zoom', 'εσσα': 'all the same', '和製漢語': 'And making Chinese', 'ऐलोपैथिक': 'Allopathic', 'создала': 'has created', 'إدمان': 'addiction', '游泳游': 'Swimming tour', '狮子头': 'Lion head', 'दिये': 'Given', 'पास': 'near', 'εντjα': 'in', 'まする': 'To worship', 'हाल': 'condition', '짱깨': 'Chin', 'জল': 'Water', 'चीज': 'thing', 'îmwe': 'one', '胡江南': 'Hu Jiangnan', 'तमाम': 'All', 'कट्टर': 'Hardcore', '魔法师': 'Magician', 'బుూ\u200c': 'Buu', 'चनचल': 'Chanchal', 'सीधा': 'Straightforward', '不要人夸颜色好': "Don't exaggerate the color", 'हिमालय': 'Himalaya', 'सिंह': 'Lion', 'خراميشو': 'Wastewater', 'мoѕт': 'the moment', 'কোথায়': 'Where?', 'نصف': 'Half', 'رائع': 'Wonderful', 'добрый': 'kind', 'ज़र्रा': 'Zarra', 'يعقوب': 'Yacoub', 'सम्झौता': 'Agreement', 'ān': 'yes', '했다': 'did', 'ἀριστοκράτης': 'aristocrat', 'çan': 'Bell', '太阳': 'sun', 'सर': 'head', 'गुरु': 'Master', '嘉定': 'Jiading', '故乡': 'home', '安倍晋三': 'Shinzo Abe', 'मच्छर': 'Mosquito', 'सभी': 'All', '成語': 'idiom', '唯我独尊': 'Only me', 'นะค่ะ': 'Yes', 'นั่นคุณกำลังทำอะไร': 'What are you doing', 'ın': 's', 'الشامبيونزليج': 'Shamballons', '抗日神剧': 'Anti-Japanese drama', '小妹': 'Little sister', 'çok': 'very', 'लोड': 'Load', 'साहित्यिक': 'Literary', 'लाल।': 'Red.', 'فى': 'in a', '絕不放過一人」之勢': 'Never let go of one person', '国家主席': 'National president', '异议': 'objection', 'अनुस्वार': 'Anuswasar', 'តើបងប្អូនមានមធ្យបាយអ្វីខ្លះដើម្បីរក': 'What are some ways to find out', '允≱ၑご搢': '≱ ≱ ≱ 搢', '黒髪': 'Black hair', '自戳双目': 'Self-poke binocular', 'βροχέως': 'rainy', 'ని': 'The', '一带一路': 'Belt and Road', 'śliwińska': 'Śliwińska', 'κατὰ': 'against', '打古人名': 'Hit the ancient name', 'குறை': 'Low', 'üjin': 'Uji', 'öppning': 'opening', 'अन्जल': 'Anjal', '이와': 'And', 'आविष्कार': 'Invention', 'غنج': 'Gnostic', 'рассвет': 'dawn', 'होना': 'Happen', 'বাড়ি': 'Home', '入定': 'Enter', 'भेड़': 'The sheep', 'देगा': 'Will give', 'ब्रह्मा': 'Brahma', 'बीन': 'Bean', 'χράω': "I'm glad", 'तीन': 'three', 'ضرار': 'Drar', 'विराजमान': 'Seated', 'सैलाब': 'Salab', 'συνηθείας': 'communication', '구경하고': 'To see', 'चुल्ल': 'Chool', '红宝书': 'Red book', '羡慕和嫉妒是不一样的。': 'Envy and jealousy are not the same.', 'मेरा': 'my', 'कलटि': 'Katiti', 'हिमाती': 'Himalayan', 'ಸಾಧು': 'Sadhu', 'عطيتها': 'Her gift', 'छान': 'Nice', 'เกาหลี': 'Korea', 'íntruments': 'instruments', 'يتحمل': 'Bear', 'रुस्तम': 'Rustam', 'बरात': 'Baraat', 'रंग': 'colour', '나이': 'age', 'פרפר': 'butterfly', '老乡': 'Hometown', '謝謝': 'Thank you', 'í‰lbƒ': '‰ in lbƒ', 'हरजाई': 'Harjai', 'পেতে': 'Get to', '「勉強': '"study', 'रड़कना': 'To cry', '清真诊所': 'Halal clinic', 'أفضل': 'Best', 'استطيع': 'I can', 'नाकतोडा': 'Nag', 'बड़ि': 'Elder', 'वैद्य': 'Vaidya', 'أنَّ': 'that', 'いたロリィis': 'There was Lory', 'ìn': 'print', '本人堅持一個中國的原則': 'I adhere to the one-China principle', 'океан': 'ocean', 'बाद': 'after', '禮儀': 'etiquette', 'सिषेविरे': 'Sisavir', 'अमृत': 'Honeydew', 'بدو': 'run', 'ὅς': 'that is', 'छोटा': 'small', 'स्वभावप्रभवैर्गुणै': 'Nature is bad', 'ענג': 'Tight', 'שלמה': 'Complete', 'डोरे': 'Dore', '我害怕得毛骨悚然': 'I am afraid of creeps.', 'झोलि': 'Jolly', 'единадесет': 'eleven', 'என்ன': 'What', 'आयुरवेद': 'Ayurveda', '堵天下悠悠之口': 'Blocking the mouth of the world', 'الجمعي': 'Collective', 'ángel': 'Angel', 'रोना': 'crying', 'हमराज़': 'Humraj', 'चिलमन': 'Drape', 'औषध': 'Medicine', 'निया': 'Nia', 'תעליעוכופור': 'Go up and go', '中国队大胜': 'Chinese team wins', 'δ##': 'd ##', 'चला': 'walked', 'पर': 'On', 'ἔργον': 'work', 'रंक': 'Rank', 'सिक्ताओ': 'Sixtao', '陳太宗': 'Chen Taizong', 'لومڤور': 'لوموور', 'விளையாட்டு': 'Sports', '的？': 'of?', '안녕하세요': 'Hi', 'נבאים': 'Prophets', 'ন্য': 'N.', 'şimdi': 'now', 'भरना': 'Fill', 'धरी': 'Clog', '漢字': 'Chinese character', 'इसि': 'This', 'आवर': 'Hour', 'काटना': 'cutting', '僕だけがいない街': 'City where I alone does not exist', 'जूठा': 'Lier', 'çekerdik': 'We Are', 'čaj': 'tea', 'నందొ': 'Nando', '核对无误': 'Check is correct', '無限': 'unlimited', 'समेटना': 'Crimp', 'żurek': 'sour soup', 'আছে': 'There are', 'مرتك': 'Committed', 'आपाद': 'Exorcism', 'चोरी': 'theft', 'బవిష్కతి': 'Baviskati', 'निकालना': 'removal', 'सीमा': 'Limit', '配信開始日': 'Distribution start date', '宝血': 'Blood', 'ग्यारह': 'Eleven', 'गरीब': 'poor', '있고': 'There', 'île': 'island', 'स्मि': 'Smile', 'енгел': 'engel', 'вы': 'you', 'οπότε': 'so', 'सूनापन': 'Desolation', 'áraszt': 'exudes', 'मारि': 'Marie', '서로를': 'To each other', 'घोपना': 'To announce', 'फितूर': 'Fitur', 'あからさまに機嫌悪いオーラ出してんのに話しかけてくるし': 'I will speak to the out-putting aura outright bad', '封柏荣': 'Feng Bairong', '「褒められたものではない」': '"It is not a praise"', '傳統文化已經失存了': 'Traditional culture has lost', '水木清华': 'Shuimu Tsinghua', 'पन्नी': 'Foil', 'ਰਾਜਬੀਰ': 'Rajbir', '煮立て': 'Boiling', '外国的月亮比较远': 'Foreign moon is far away', 'شغف': 'passion', 'గురించి': 'About', 'övp': 'ÖVP', '实名买票制': 'Real name ticket system', 'बिखरना': 'To scatter', 'التَعَمّقِ': 'Persecution', 'जीवें': 'The living', 'गई': 'Has been', '部頭合約': 'Head contract', 'دقیانوس': 'Precise', 'फंकी': 'Funky', '粵拼': 'Cantonese fight', 'ĉiohelpanto': 'all supporter', 'मारना': 'killing', 'मानजा': 'Manja', 'अष्टांगिक': 'Octagonal', 'への': 'To', 'रहना': 'live', 'खाना': 'food', 'ಕೋಕಿಲ': 'Kokila', 'చిన్న': 'Small', '金髪': 'Blond hair', '籍贯': 'Birthplace', 'उखाड़ना': 'Extirpate', 'α2': "A'2", 'להשתלט': 'take over control', '露西亚': 'Lucia', '大有「寧可錯殺一千': 'There is a lot of "I would rather kill a thousand', 'संस्कृत': 'Sanskrit', 'христо': 'Christo', 'लगाना': 'to set', '선배님': 'Seniors', 'उड़ीसा': 'Orissa', 'ஆய்த': 'Ayta', 'तेरे': 'Your', 'आकांक्षी': 'Aspiring', 'बाज़ार': 'The market', 'हर्ज़': 'Herz', 'だします': 'Will do.', 'चक्कर': 'affair', '없어': 'no', 'चम्पक': 'Champak', 'ताल': 'rythm', 'āgamas': 'hobby', 'योद्धा': 'Warrior', 'αλυπος': 'chain', 'बेड़ा': 'Fleet', 'बात': 'talk', '做莫你酱紫的？': 'Do you make your sauce purple?', '見逃し': 'Miss', '鰹節': 'Bonito', 'तथैव': 'Fact', 'आध्यात्मिकता': 'Spirituality', '正弦': 'Sine', 'लिया': 'took', 'îndrăgire': 'love', '我願意傾聽': 'I am willing to listen', '家乡': 'hometown', '大败': 'Big defeat', 'मए': 'Ma', 'జ్ఞ\u200cా': 'Sign', 'օօ': 'oo', 'खेमे': 'Camps', 'الشوكولاه': 'Chocolate', 'полностью': 'completely', '商品発売日': 'Product Release Date', '金継ぎ': 'Gold piecing', '烤全羊是多少人民币呢？': 'How much is the renminbi roasted in the whole sheep?', 'γолемата': 'golem', '孫瀛枚': 'Grandson', 'özlüyorum': 'I am missing', 'خلبوص': 'Libs', 'アンダージー': 'Undergar', '蝴蝶': 'butterfly', 'প্রশ্ন': 'The question', 'जरूरी': 'Necessary', 'बूरा': 'Bura', '有毒': 'poisonous', 'सान्त्वना': 'Comfort', '눈치': 'tact', '\ufeffwhat': 'what', 'çeşme': 'fountain', 'ごっめんなさい': 'Please, sorry.', 'விளக்கம்': 'Description', '罗西亚': 'Rosia', 'गोटा': 'Knot', 'लेखावलिपुस्तके': 'Accounting Tips', 'सम्भोग': 'Sexual intercourse', '慢走': 'Slow walking', 'तुम': 'you', 'लीला': 'Leela', 'ฉันจะทำให้ดีที่สุด': 'I will do my best', 'चम्पु': 'Shampoo', '視聽華語': 'Audiovisual Chinese', '比比皆是': 'Abound', 'धावा': 'Run', '娘娘': 'Goddess', 'पहाड़': 'the mountain', 'राजा': 'King', '茹西亚': 'Rusia', 'ब्राह्मणक्षत्रियविशां': 'Brahmin Constellations', '수업하니까': "I'm in class.", 'చెల్లెలు': 'Sister', 'साजे': 'Made', '支那人': 'Zhina', '团表': 'Group table', '讲得': 'Speak', 'へからねでて': 'From to', 'über': 'over', 'քʀɛʋɛռt': 'kʀɛʋɛr t', 'นเรศวร': 'Naresuan', '方言': 'dialect', 'पना': 'Find out', '怕樱': 'Afraid of cherry', 'घुल': 'Dissolve', '米帝': 'Midi', 'طيب': 'Ok', 'प्रेम': 'Love', 'पढ़ाई': 'studies', '他穿褲子': 'He wears pants', 'मेरी': 'Mine', 'উত্তর': 'Reply', 'स्थिति': 'Event', '\x1b\xadü': '?\xadü', 'तैश': 'Tachsh', '写得好': 'Well written', 'मॉलूम': 'Known', '创意梦工坊': 'Creative Dream Workshop', 'आपकी': 'your', 'मिलना': 'Get', '배웠어요': 'I learned it', '許自東': 'Xu Zidong', 'जाऊँ': 'Go', 'अहिंसा': 'Nonviolence', 'джоу': 'Joe', '金繕い': 'Gold patting', '好心没好报': 'No return on a good deed', 'çevapsiz': 'unanswered', 'मिल': 'The mill', '日本語': 'Japanese', 'اخذ': 'Get', '水军': 'Water army', 'बिना': 'without', 'बनाना': 'Make', 'التوبه': 'Repentance', '一个灵运行在我的家': 'a spirit running in my home', '한국어를': 'Korean', 'कि': 'That', 'εισα': 'import', 'लगना': 'feel', 'गपोड़िया': 'Gopodiya', 'ārūpyadhātu': 'exhyadadate', 'आए': 'Returns', '吃好吃金': 'Eat delicious gold', 'นั่น': 'that', 'बढ़ाना': 'raise up', '보니': 'Bonnie', '爱着': 'Love', '선배': 'Elder', '刷屏': 'Brush screen', '人性化': 'Humanize', 'خرسانة': 'Concrete', '麻辣乾鍋': 'Spicy dry pot', '\x10œø': '? œø', 'सतरंगी': 'Satarangi', '磨合': 'Run-in', 'को': 'To', 'شهادة': 'Degree', 'একটি': 'A', 'へと': 'To', 'يلي': 'Following', '光本': 'Light source', '褒め殺し': 'Praise kill', 'आँचल': 'Anchal', 'এর': 'Of it', 'पिटा': 'Pita', 'بديش': 'Badish', 'गंगु': 'Gangu', '可是': 'but', 'की': 'Of', '谢谢。台灣同胞': 'Thank you. Taiwan compatriots', 'दम': 'power', 'मैकशि': 'Macshi', 'రాజ': 'King', '玉蘭花': 'Magnolia', '江ノ島盾子': 'Enoshima Junko', 'ѕтυpιd': 'ѕtυpιd', 'जिसका': 'Whose', 'எழுத்து': 'Letter', '甲骨文': 'Oracle', 'चूरमा': 'Churma', 'चूलें': 'Chulen', 'प्रविभक्तानि': 'Interpretation', 'いる': 'To have', 'مقال': 'article', 'पाय': 'Feet', 'अतीत': 'Past', 'ármin': 'Armin', '東夷': 'Dongyi', 'आदमकद': 'Life expectancy', 'किये': 'Done', 'पतवार': 'Helm', '楽曲派アイドル': 'Song musical idols', 'डगमगाना': 'Waver', '북한': 'North Korea', '禮記': 'Book of Rites', '西魏': 'Xi Wei', '過労死': 'Death from overwork', 'बेबुनियाद': 'Unbounded', '仙人跳': 'Fairy jump', '港女': 'Hong Kong girl', '虞朝': 'Sui Dynasty', 'µ0': 'μ0', '字母词': 'Letter word', 'अर्धांगिनि': 'Arghangini', '真功夫': 'real Kong Fu', '飯糰': 'Rice ball', 'علم': 'Science', 'गुड़': 'Jaggery', 'гречку': 'buckwheat', '我方记账数字与贵方一致': 'Our billing figures are consistent with yours', 'आने': 'To arrive', 'कब': 'When', '宋楚瑜': 'James Soong', 'διητ': 'filter', 'राई': 'Rye', 'விரதம்': 'Fasting', 'बदल': 'change', '成语': 'idiom', 'ĺj': 'junk', 'какая': 'which one', '文翰': 'Wen Han', 'کـ': 'K', 'ʀɛċօʍʍɛռɖɛɖ': 'ʀɛċ oʍʍɛрɖɛɖ', 'दिलों': 'Hearts', '星期七': 'Sunday', '傳送': 'Transfer', '陳云根': 'Chen Yunen', 'ʿalaʾ': 'Allah', 'गुल': 'Gul', 'ख़बर': 'The news', 'मन्च': 'Manch', '国家知识产权局': 'State Intellectual Property Office', '행복하게': 'happily', 'बबीता': 'Babita', 'юродивый': 'holy fool', 'सफाई': 'clean', 'вода': 'water', 'लाठि': 'End', 'γὰρ': 'γσρ', '在日朝鮮人／韓国人': 'Koreans/Koreans in Japan', '学过': 'Learned', 'जमाना': 'Solidification', 'इकोनॉमिक्स': 'Economics', 'क्या': 'what', 'ドア': 'door', '中国的存在本身就是原罪': 'The existence of China itself is the original sin', 'मौसमि': 'Season', 'ठाठ': 'Chic', 'ἡμετέραν': 'another', '火了mean': 'Fired mean', 'せえの': 'Set out', 'ஆயுத': 'Arms', 'कंपनी': 'company', 'अहम्': 'Ego', 'भर': 'Filled', 'फिरना': 'To revolve', '高山族': 'Gaoshan', '王飞飞是一个哑巴的婊子。': 'Wang Feifei is a dumb nephew.', '反清復明': 'Anti-clearing', '关门放狗': 'Close the dog', '中国民族伟大复兴？i': 'The great revival of the Chinese nation? i', '漢名': 'Han name', 'ín': 'into the', '이름은': 'name is', '뽀비엠퍼러': 'Pobi Emperor', '堅決反對台獨言論': 'Resolutely oppose Taiwan independence speech', 'ड़': 'D', 'ادب': 'Literary', '〖2x〗': '〖〗 2x', 'في': 'in a', 'परन्तप': 'Parantap', '不正常人類研究中心': 'Abnormal human research center', 'метью': 'by the way', 'σε': 'in', '罗马炮架': 'Roman gun mount', 'đỗ': 'parked', '二哈': 'Erha', '中國': 'China', 'मुझे': 'me', 'бджа': 'bzha', 'छुटाना': 'To leave', 'সহ': 'Including', 'δίδυμος': 'twin', 'दौरा': 'Tour', 'आया': 'He Came', 'ľor': 'Lor', 'شاف': 'balmy', 'افشلك': 'I miss you', 'पता': 'Address', 'śląska': 'Silesian', '我还有几个条件呢。': 'I still have a few conditions.', '元寇': 'Yuan', 'सहायक': 'Assistant', 'टाइम': 'Time', 'समुदाय': 'Community', 'टिपटिपवा': 'Tiptipava', '五毛党': 'Wu Mao Party', 'दे': 'Give', '屠城': 'Slaughter city', 'कहने': 'To say', 'कलमूहा': 'Kalamuha', 'בפנים': 'in', 'कर्माणि': 'Creation', 'अथवा': 'Or', 'રાજ્ય': 'State', 'घसोना': 'Shed', '今そう言う冗談で笑える気分じゃないから一人にしてって言ったら何があったの？話聞くよって言われたんだけど': 'I do not feel like being laughing with what I say now, so what happened when I told you to be alone? I was told by listening and talking', 'परमो': 'Paramo', 'υφηιρειτω': 'I suppose', 'כתובים': 'Written', '能夠': 'were able', '고등학교는': 'High School', 'बजरि': 'Breeze', 'जानेदिल': 'Knowingly', '大胜': 'Big win', 'काल': 'period', 'すみません': "I'm sorry", 'हिमाकत': 'Snow plant', 'þîfû': 'iphone', 'よね': 'right', 'هالنحس': 'Halting', 'الحجازية': 'Hijaz', 'жизнь': 'a life', 'किया': 'Did', '沸騰する': 'To boil', 'นะครับ': 'Yes', 'پیش': 'Before', 'परदाफास': 'Bustard', 'वफाएं': 'Affection', 'सैनानि': 'Sanani', 'ölü': 'dead', 'हैं': 'Are', '宋美齡': 'Song Meiling', 'கவாடம்': 'Valve', '我是一名来自大陆的高中生': 'I am a high school student from the mainland.', 'اجمل': 'The most beautiful', '가용': 'Available', 'चरम': 'Extreme', '象形文字': 'Hieroglyphics', '男女授受不亲': 'Men and women don’t kiss', 'моя': 'my', '疑义': 'Doubt', '管中閔': 'In the tube', 'çekseydin': 'pulls you were', 'عم': 'uncle', 'ルージュ': 'Rouge', '永遠': 'forever and always', 'بيت': 'a house', '「褒められた」': '"I was praised"', 'कर्णजित्': 'Karnajit', 'あたまわるい': 'Bad headache', 'ε0': 'e0', 'şafak': 'dawn', 'जलाओ': 'Burn', '観世音菩薩': '観世音', 'पीत': 'Yellow', 'दारी': 'Dari', 'የሚያየኝን': 'What i see', 'هاپو': 'Dog', '发声点': 'Vocal point', 'être': 'be', 'மாண்பாளன்': 'Manpalan', '감겨드려유': 'You can wrap it.', '知音': 'Bosom friend', 'एक': 'One', '\x01jhó': '? Jho', 'かんぜおんぼさつ': 'Punctuation', 'ødegaard': 'Ødegaard', '得理也要让人': 'It’s ok to make people', 'مرة': 'Once', 'दो': 'two', 'ने': 'has', '用乡村包围城市': 'Surround the city with the countryside', 'مكتب': 'Office', 'řepa': 'beet', 'день': 'day', '江青': 'Jiang Qing', 'ángeles': 'angels', 'çonstant': 'constant', 'टिपटिपुआ': 'Tip Tip', 'ஆன்லைன்': 'Online', '盧麗安': "Lu Li'an", 'самый': 'most', 'からかってくる男に': 'To the coming men', 'بعرف': 'I know', '知乎': 'Know almost', 'पहेलि': 'Puzzles', 'बहार': 'spring', 'भंगिमाँ': 'Bhangima', 'くわんぜおんぼさつ': 'Honey', '河殤': 'River', 'شو': 'Shaw', 'محمد': 'Mohammed', 'спереди': 'in front', '没毛病': 'No problem', 'árbenz': 'Arbenz', 'लार्वा': 'Larvae', 'चढ़ा': 'Ascend', 'तो': 'so', 'يلعب': 'Play', 'घिसा': 'Ginger', 'रेखागड़ित': 'Sketchy', 'मुरदा': 'Murada', 'चित्र': 'picture', 'தாக்கல்': 'Filing', '大败美国队': 'Big defeat to the US team', 'искусственный': 'artificial', 'चाल': 'Trick', 'घुटता': 'Kneeling', 'बिगड़ना': 'Deteriorate', '您这口味奇特也就罢了': 'Your taste is strange.', 'लायक': 'Worth', '大篆': 'Daxie', 'खिलाना': 'To feed', 'ماهو': 'What is the', 'पहुँचनेके': 'To reach', '外来語': 'Foreign language', 'चटका': 'Click', 'کوالا': 'Koala bear', '不过': 'but', 'पड़ना': 'Fall', 'पानी': 'Water', '저의': 'my', '一生懸命': 'Hard', 'مش': 'not', 'लिखामि': 'Written', 'गया': 'Gaya', 'компания': 'company', 'तेरि': 'Teri', '茶髪': 'Brown hair', '하다': 'Do', '不是共和国；中华人民共和国和朝鲜民主主义人民共和国是共和国': 'Not a republic; the People’s Republic of China and the Democratic People’s Republic of Korea are Republic', '烤全羊多少人民币呢？': 'How much is the price of roast whole sheep?', 'घर': 'Home', 'पाला': 'Frost', '左右手': 'Left and right hand', 'کوالالامپور': 'Kuala Lumpur', 'āsanas': 'asanas', 'ոո': 'e', 'сегодня': 'Today', '저는': 'I am', 'русофил': 'blossoming', '甘蓝': 'Cabbage', '流浪到淡水': 'Wandering to fresh water', 'единайсет': 'eleven', 'परात': 'Underwear', '\x7fhow': '?how', '¡que': 'what', 'ἀχιλλεύς': 'Achilles', 'так': 'So', 'آداب': 'Rituals', '\x10i': '?i', 'मट्ठा': 'Whey', '平天下悠悠之口': 'The mouth of the world', 'लुन्डा': 'Lunda', 'ся': 'camping', 'вареники': 'Vareniks', 'δο': 'gt;', 'öyle': 'so', 'पुरे': 'Enough', '看他不顺眼': 'Seeing him not pleasing to the eye', '「o」': '"O"', 'とする': 'To', 'হচ্ছে': 'Being', 'लाखों': 'Millions', 'α1': "A'1", 'نهائي': 'Final', 'ɾ̃': 'ɾ', 'तिलक': 'Tilak', 'لا': 'No', 'صور': 'photo', '怒怼': 'Roar', 'மௌன': 'Mauna', 'परिस्थिति': 'Situation', '서로가': 'Mutually', '山進': 'Shanjin', '蝴蝶蛋': 'Butterfly egg', 'ксш': 'ksş', '饭可以乱吃': 'Rice can be eaten', '渡します': 'I will hand it over.', 'österreich': 'Austria', 'øÿ\x13': 'øÿ?', 'ممتاز': 'Excellent', '蝴蝶卵': 'Butterfly egg', 'ही': 'Only', 'ठोकरे': 'Knock', '湿婆': 'Shiva', 'обожаю': 'love', 'लस्सी': 'Lassi', '操你妈': 'Fuck your mother', 'फौलादि': 'Fauladi', 'жизни': 'of life', 'đổi': 'change', '阻天下悠悠之口': 'Block the mouth of the world', 'オメエだよオメエと話してんのが苦痛なんだよ。シネ。': "It's omn it is painful to talk to Oume. Cine.", 'खाट': 'The cot', '译文': 'Translation', 'सँवरना': 'To embellish', 'дванадесет': 'twelve', '陳雲': 'Chen Yun', 'дванайсет': 'twelve', 'आँखें': 'Eyes', 'पूछते': 'Inquires', 'भंगी': 'Posture', 'सूना': 'Deserted', 'प्याला': 'Cup', 'には': 'To', 'доктора': 'the doctors', 'देना': 'give', 'बिरेन्र्द': 'Forget', 'गला': 'throat', 'रखा': 'Kept', 'हाले': 'Haley', '欢迎入坑': 'Welcome to the pit', 'डलि': 'Dallie', 'ôš': 'OS', 'يوسف': 'Yousuf', 'छोंकना': 'Strain', 'пп': 'pp', 'उसपर': 'on that', 'υρολογιστών': 'computers', '新年快乐！学业进步！身体健康！谢谢您们读我的翻译篇章': 'happy New Year! Academic progress! Healthy body! Thank you for reading my translation chapter.', '人民': 'people', 'घंटे': 'Hours', 'شباط': 'February', '食べる': 'eat', 'صلاح': 'Salah', '土澳': 'Tuao', '干嘛天天跟我说韩语': 'Why do you speak Korean with me every day?', 'ōnogi': 'climate', 'صاحب': 'owner', 'اكل': 'ate', '大唐': 'Datang', 'مطالعه': 'Study', '养生': 'Health', '车子': 'Car', 'कचरा': 'Garbage', 'महरबानि': 'Mehraban', 'शुष्टि': 'Shutti', 'интеллект': 'intelligence', '阮鏐': '阮镠', '鲁玥': 'Reckless', '입니다': 'is', 'ῥιζὤματα': 'Threads', 'люблю': 'love', 'ɛxɛʀċɨsɛ': 'it is not', 'कपड़े': 'dresses', 'उड़न': 'Flying', '广电总局': 'SARFT', '骂人': 'curse', 'የየየኝን': 'What do i say', 'बेलना': 'Crib', 'பல்லாக்குப்': 'Pallakkup', 'बराबर': 'equal', 'ظرف': 'Dish', 'होनोपैथिक': 'Homeopathic', '君子': 'Gentleman', '河和湖': 'Kawahata lake', '精一杯': 'Utmost', 'है': 'is', '非要以此为依据对人家批判一番': 'I have to criticize others on this basis.', 'வச்சன்': 'Vaccan', 'நான்': 'I', 'का': 'Of', '三味線': 'Shamisen', 'šwejk': 'švejk', 'дурак': 'fool', '风琴': 'organ', 'हिंसा': 'Violence', 'βιον': 'bio', 'नारी': 'Woman', '知らない': 'Do not know', 'मुँह': 'The mouth', 'अपरम्पार': 'Unperturbed', '秋季新款': 'Autumn new style', 'আমার': 'Me', 'عبقري': 'genius', 'आहें': 'Ah', 'ਨਾਮ': 'Name', 'महफ़िल': 'Mehfil', 'बटेर': 'Quail', '林彪': 'Lin Wei', 'जाने': 'Know', 'डोंगरिचाल': 'Mountain move', 'εἰρήνη': 'Irene', 'प्रतिलेखनम्': 'Transcript', 'дп': 'dp', 'उसने': 'He', '몇시간': 'how many hours', 'नैया': 'Naiya', 'ἐξήλλακτο': 'inexpensive', '彩蛋': 'Egg', 'उलटफेर': 'Reverse', '台湾最美的风景是人': 'The most beautiful scenery in Taiwan is people.', '馄饨': 'ravioli', 'सदके': 'Shake', '饺子': 'Dumplings', 'भूमिका': 'role', '为什么说': 'Why do you say', '요즘': 'Nowadays', 'गिरि': 'Giri', '中國話': 'Chinese words', 'द्रोहि': 'Drohi', '中庸之道': 'The doctrine of the mean', 'хочу': 'want', 'ḵarasāna': 'ḵarasana', '走gǒ': 'Go gǒ', 'जमाल': 'Jamal', 'मन': 'The mind', 'तेलि': 'Oilseed', '非诚勿扰': 'You Are the One', 'атом': 'atom', '中华民国和大韩民国是民国': 'The Republic of China and the Republic of Korea are the Republic of China', 'नलि': 'Nile', 'हाथ': 'hand', 'खाजला': 'Itching', 'ōe': 'yes', '것이다': 'will be', 'şoųl': 'şoùl', 'तश्तरि': 'Cleverness', 'χρῆσιν': 'use', '民族': 'Nationality', 'الرياضيات': 'Mathematics', 'にじゅうさい': 'Twelve months', 'ὠς': 'as', '恋に落ちないからよく悲しい': 'It is often sad because it does not fall in love', 'ਸ਼ੀਂਹ': 'Lion', '黎氏玉英': 'Li Shiyuying', 'فبراير': 'February', '白濮': 'Chalk', 'অধীনে': 'Under', 'प्रशंसा': 'appreciation', 'ขอพระเจ้าอยู่ด้วย': 'May God be with you', 'अडला': 'Bent', 'ده': 'Ten', 'पसीजना': 'Exudate', 'कोई': 'someone', 'கருக்குமட்டை': 'Karukkumattai', 'कन्नी': 'Kanni', 'यात्रा': 'journey', '白酒': 'Liquor', 'τὴν': 't', 'करना': 'do', 'उल': 'ul', '俄罗斯': 'Russia', 'உனக்கு': 'You', 'जौहर': 'Johar', 'ಸ್ವರಕ್ಷರಗಳು': 'Self-defense', '论语': 'Analects', 'šakotis': 'branching', '儿臣惶恐': 'Childier fear', '讲的？': 'Said?', 'প্রতিনিয়ত': 'Every day', 'சன்னல்': 'Sill', '蠢的像猪一样': 'Stupid like a pig', 'رح': 'Please', 'بس': 'Yes', 'εὔιδον': 'you see', 'सदबुद्धि': 'Good sense', 'भगवान': 'God', '사랑해': 'I love you', 'בתוך': 'Inside', 'čeferin': 'чеферин', '民国': 'Republic of China', '但是': 'but', 'सकता': 'can', 'घाट': 'Wharf', 'čechy': 'Bohemia', '抹黑': 'Smear', 'γλαυκῶπις': 'greyhounds', 'నీకెందుకు': 'Nikenduku', 'चमत्कार': 'Miracle', 'दुनिया': 'world', 'یہاں': 'Here', 'اللي': 'Elly', 'খামার': 'The farm', '一呼百诺': 'One call', 'വിഡ്ഢി': 'Stupid', 'दिल': 'heart', 'тeenage': 'trinage', '皇上': 'emperor', 'टटोलना': 'Grope', '犬子': 'Dog', '我希望有一天你沒有公王病': 'I hope that one day you don’t have a king', '并无分裂中国的意图': 'No intention to split China', 'óscar': 'oscar', 'ኤልሮኢ': 'Alright', 'ŷhat': 'hhat', '천사': 'Angel', 'दी।': 'Given', 'रड़क': 'Raze', 'कानी': 'Kani', '江之島盾子': 'Enokima Shiko', '老生常谈': 'Old talk', 'των': 'of', '星期日': 'on Sunday', 'पैन्डा': 'Panda', 'マリも仲直りしました': 'Mari also made up.', 'ты': 'you', 'देश': 'Country', 'ठंडक': 'Coolness', 'নামল': 'Get down', 'जर्मनी': 'Germany', 'шли': 'walked', 'たべる': 'To eat', 'लिए': 'for', '鸡汤文': 'Chicken soup', 'มวยไทย': 'Thai boxing', '簡訊': 'Newsletter', 'منزل': 'Home', 'कर': 'Tax', '生女眞': 'Daughter-in-law', 'ек': 'ek', 'સંઘ': 'Union', '\u200bsalarpuria': 'Salphary', 'ţara': 'the country', 'नही': 'No', 'मगज': 'Mercury', 'অক্ষয়': 'Akshay', 'حال': 'Now', 'चिन्दी': 'Chindee', 'τῆς': 'her', '福哒柄': 'Good fortune', '청하': 'Qinghai', '越人': 'Yueren', 'なかなかに謎だな': "It's quite a mystery.", 'रखना': 'keep', 'பரை': 'Parai', 'करके': 'By doing', 'فِي': 'F', 'गुथना': 'Knit', '话不可以乱讲': "Can't talk nonsense", 'वैकल्पिक': 'Alternative', 'ਨਾਮੁ': 'Name', '你别高兴得太早': "Don't be too happy too early", '煎餅': 'Pancake', '한다': 'do', 'सबब': 'Cause', 'বিষয়টি': 'Matter', 'कोसना': 'To crack', 'ㅜㅜ': 'ㅜ', 'অক্সয়': 'Akshay', 'الدوالي': 'Varicose veins', 'பயிர்ப்பு': 'Yields', 'अजा': 'SC', 'あの色々': 'That kind of variety', 'емеля': 'emel', 'मेवा': 'Meva', 'जलवाफरोज़': 'Jalwa Phoroz', '中庸': 'Moderate', 'उसके': 'his', 'अहम': 'Important', 'वहम': 'Vanity', 'ís': 'ice', 'कलात्मक': 'Artistic', 'ἀχιλῆος': 'Achilles', '民族罪人': 'National sinner', 'मुकाम': 'Peer', 'ão': 'to', '한국': 'Korea', 'ادهر': 'Idir', '一長': 'One long', 'てくれませんか': 'Would you please', 'ブレイク': 'break', 'शहनाई': 'the clarinet', 'तीरन्दाज़ि': 'Arrows', 'रूढ़ीवादि': 'Conservative', 'झाँकी': 'Peeping', 'सत्यवादी': 'Truthful', '郑琳': 'Zheng Lin', 'युनानी': 'Unani', 'φώνας': 'light', 'जमना': 'Solidify', '〖plg〗': '〖Plg〗', 'हरी': 'Green', 'بطولة': 'championship', '这些词怎么读？这些词怎么说？这些词怎么念？which': 'How do you read these words? What do these words say? How do you read these words? Which', 'алтерман': 'alterman', 'اَلبَحْثِ': 'البحث', 'موجود': 'Available', 'कश्ति': 'Power', 'اسم': 'Noun', 'लाज़मि': 'Shameful', 'तुमने': 'you', '공화국': 'republic', 'लुक्का': 'Lukka', '史记': 'Historical record', '포경수술': 'Circumcised', '高端大气上档次into': 'High-end atmospheric grade into', 'что': 'what', 'योध': 'Rift', 'धर्म': 'religion', 'दरदर': 'Tariff', '訓読み': 'Kun Readings', 'впереди': 'in front', '민국': 'Republic of Korea', 'εντσα': 'in', '我搜了这本小说': 'I searched this novel.', '杨皎滢？': 'Yang Wei?', 'भारतीयों': 'Indians', '巴蜀': 'Bayu', '\x02tñ\x7f': '? tñ?', 'αβtαβ': 'aba', 'जल': 'water', 'बाध्य': 'Bound', 'মহাবিশ্ব': 'Universe', 'প্রসারিত': 'Stretch', 'अन्जाम': 'Anjaam', 'जीतना': 'win', 'कड़ा': 'Hard', '刁民': 'Untouchable', 'ขอให้พระเจ้าอยู่ด้วย': 'May God live too.', '油腻': 'Greasy', 'ᗯoᗰeᑎ': 'ᗯoᗰe ᑎ', 'להתראות': 'Goodbye', 'वाले': 'Ones', 'አየሁ': 'I saw', 'ओर': 'And', 'ずand': 'Without', 'निगोड़ा': 'Nigoda', 'эй': 'Hey'}
#print(words_to_english_map)

def _get_unknown_word(unknown_dict):
    unknown_re = re.compile('(%s)' % '|'.join(unknown_dict.keys()))
    return unknown_dict, unknown_re

unknown_dict, unknown_re = _get_unknown_word(words_to_english_map)
def replace_typical_unknown(text):
    def replace(match):
        return unknown_dict[match.group(0)]
    return unknown_re.sub(replace, text)

# Clean unknown words
train["question_text"] = train["question_text"].apply(lambda x: replace_typical_unknown(x))
test["question_text"] = test["question_text"].apply(lambda x: replace_typical_unknown(x))
