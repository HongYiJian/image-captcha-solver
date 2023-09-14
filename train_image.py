import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import pandas as pd
from torchvision.transforms import Compose, ToTensor
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.nn as nn
import time
import torch.nn.functional as F

ans_text = '0123456789' # 可能出現文字
ans_length = 10 # 可能出現文字數量 ans_text.length
ans_text_length = 6 # 圖片文字數量
batch_size = 25 # 訓練效率，越大越吃電腦效能，但也訓練越快
base_lr = 0.6 # 學習率，如需調整需上網參考應設定值
max_epoch = 60 # 最多訓練次數 epoch，可提前用 Ctrl + C 中斷
model_path = './tset_densenet.pth' # 輸出模型檔

# 訓練圖檔答案
ans_list = ['532332','757105','505806','733086','117608','968027','770375','509831','680107','667580','869386','462875','687624','800667','613795','897283','584887','545201','569136','456049','201808','241462','704914','556733','000518','338400','312955','856697','368908','547473','243555','384179','294302','784380','706645','606141','618599','246464','821614','778241','980507','137169','801952','695465','926054','802819','426175','125636','325598','415689','725361','670181','513485','066136','408448','681862','269904','769143','480906','286344','960454','828450','657050','056889','725739','626801','758656','850206','688863','820408','693101','854796','396640','873543','756784','076019','056287','805344','292731','909644','151362','514672','502229','947128','596230','354904','612528','402435','212928','353506','129066','784816','617566','309207','858986','992228','901709','748094','566550','765622','695073','758794','978446','379261','620373','794442','406227','237787','785915','458668','862074','454770','175701','924015','721003','901088','306946','113816','213254','287742','570836','288531','894510','194341','535682','680696','473994','183632','511681','367738','889318','722390','712273','375448','081109','175663','401723','070275','268651','940598','046538','376255','642511','584505','995116','859562','939720','420696','571237','386286','742984','709370','436441','816703','297192','774009','989463','484508','107582','367226','275507','610022','564076','455871','958770','087241','378060','416702','087522','796598','897321','440607','087134','171726','730148','309241','936982','696734','989201','246190','004544','702762','471014','066534','536271','975718','614900','186228','503836','206967','512208','758224','680881','422790','714511','556233','966381','174812','238403','515616','590128','529469','782813','319190','869092','575796','085491','766419','651254','074551','048016','568598','587695','071112','586938','468238','811201','278515','596336','165484','384529','318303','122463','174097','831858','529388','041521','032073','251723','271667','659096','385435','223501','991126','050601','987124','369310','584439','709748','691189','427710','735599','152327','134069','366592','295425','301891','731752','108047','091160','265858','571499','704295','614900','012963','398026','109922','635184','165945','930891','026702','093841','924431','815935','810478','684301','766538','300390','673622','176764','354017','778618','392863','703876','529188','714823','270305','080725','248514','717177','128569','761800','376740','712805','211159','201067','766887','040752','487579','887702','462545','812303','582748','634151','764579','132000','106208','602744','569783','501702','054586','047681','116314','386440','309649','734848','029945','150515','205951','148010','292897','650578','183157','610358','838859','723353','184575','042684','706697','760726','154946','268432','072447','328678','992677','977981','705558','324746','492855','339821','079476','460876','780907','484931','160625','465421','082728','734236','265391','915516','663417','153359','209980','426292','035024','317920','520743','871391','406650','310937','298892','744304','522034','757940','759531','661453','191542','576158','795245','006370','325411','582597','258500','746576','930102','949902','681047','358115','096020','459738','181688','385153','603983','948401','327123','530339','715410','736439','958828','677607','728315','818378','926401','076703','039266','380910','643990','897544','994733','328267','477824','436441','404509','820681','334076','849273','319441','404350','123405','747139','037616','549302','332185','912194','135724','478252','900976','729115','395817','594592','416138','570755','826305','377573','562542','025962','753179','209244','319073','542480','665410','626209','084018','561076','698557','961548','530575','131513','592953','035162','343857','322906','061479','683035','694831','712854','944307','684415','160044','947331','029152','999731','878367','635774','768224','847190','422323','396203','930691','648677','865601','696004','406276','892173','545318','293549','024926','812568','521677','684837','734411','305532','052578','387607','068875','272475','449100','903532','517604','074716','589814','120745','091324','515338','080780','042502','101940','781862','376499','465621','053167','359867','353892','135724','540922','839181','682170','327597','007636','007385','026170','981483','398210','195261','730582','387003','112399','668128','046425','669026','543338','576151','942695','823302','392480','949115','693677','007753','382429','384471','752167','993470','858986','998274','631107','512546','497965','333891','825429','000721','774720','152165','335431','207186','325343','195931','248070','461388','467068','389100','403018','350333','207825','677793','621124','784050','902664','458411','438781','987176','901239','961140','145137','987087','192794','852039','600035','337731','703855','594824','784050','387140','969228','655557','545833','445524','074940','774977','191416','469243','681797','890973','557020','875258','668226','867297','636897','585819','015846','131651','926601','688231','935883','887263','977108','773650','165801','760330','176977','716958','376919','996322','967046','102320','143470','038346','168618','714012','882789','819359','090423','932492','229186','604129','715094','906522','466365','418076','031802','922210','511770','781308','729927','078288','674517','442734','092708','619837','363170','080214','615997','643630','446907','418352','493905','412604','559583','344910','473877','396270','267728','867121','424167','068242','562273','513668','713106','609539','191741','992015','796696','151943','018865','427515','343905','486448','232301','484062','251554','263930','493286','435052','911269','975441','519484','066225','021740','038044','158387','988130','618217','860317','309068','749779','858810','669281','288408','325466','413922','381458','249784','384268','965873','928535','991665','703127','740421','493431','384636','329467','156490','497267','085284','887534','572737','116710','832152','303063','696657','184025','177983','729181','499973','545929','168910','465418','747588','977245','600458','752954','488320','023543','297357','382677','157172','090006','812001','170488','432116','607999','777636','089056','012137','795863','834400','828781','802517','761229','953340','931710','254625','545465','719865','965656','517906','519312','065518','494089','180652','931157','791975','397206','652418','633557','840959','295387','387075','121447','101521','706631','858054','843133','953837','394309','312742','411117','489203','104173','503499','147406','660895','849088','691659','642603','398871','851446','286712','265769','592116','849717','391861','596638','654330','551707','417744','956132','324109','133803','564281','947021','080358','843698','932314','524329','557490','814153','770592','798897','571986','802685','937801','064071','046146','505273','823887','755341','436354','407865','963318','103006','842008','288026','860592','224382','033656','246596','195474','382239','768805','908192','495751','378913','322353','667301','504110','762627','798302','685792','053820','613795','446554','917743','485933','744690','259705','696268','201795','609325','686005','144335','547085','091043','471014','119228','911046','567886','230892','071531','125055','693220','570551','028601','510562','519391','151857','835465','105457','963638','290395','423840','540176','461124','410854','284219','750878','513746','263910','069963','033773','963154','841875','569787','152347','941689','180969','041153','723566','892960','089121','070488','507659','775366','837960','474444','135938','627490','857135','635107','317765','404941','279282','322130','785062','720750','423074','476179','779996','068474','766899','664560','845278','657063','068407','961185','212763','019173','745519','645795','238374','383465','253363','925446','721384','505487','492439','856229','786849','346977','433004','639072','016576','566083','899702','640702','463764','064321','350133','068456','985361','037616','147437','403410','635627','273441','745325','168511','423221','543513','237416','336956','343255','913346','833009','376619','782125','781326','751997','363823','064499','861208','624212','167372','573575','492312','660602','127776','919174','780575','031346','077692','947052','500445','851160','669446','447892','239575','546533','931144','914579','777287','146441','299695','011515','689779','428746','566278','676825','286345','350184','247399','005346','782078','880079','572222','985203','797785','709569','309839','740548','194808','644944','647878','479306','408664','711735','207924','510114','096779','164816','332292','496347','319915','265346','175752','093474','459825','612744','602881','876281','160742','668128','621687','944562','242293','449344','086290','450812','832695','699126','803993','506536','464347','956830','079874','446907','692064','395751','701587','392645','294323','478788','982778','734491','760359','474879','332185','667212','286207','693984','465514','822184','619584','426960','653324','489254','466454','459127','196479','513804','173957','605609','860104','755554','448489','110700','836391','295689','292420','793361','997537','742933','073402','893237','239419','854981','599213','375953','032935','379306','915740','953322','343586','586612','947778','561742','684116','901913','930474','288827','946233','841900','066641','346138','779624','632455','052251','194925','323774','814836','021810','536054','327484','615170','894861','860677','255831','304520','028896','481486','574230','212131','864460','573973','346973','230906','870124','750878','713738','540680','115340','234231','327683','509295','979126','087134','223377','765036','212395','675017','395264','366424','148789','265810','452528','780375','302226','349948','136406','725063','138288','595623','975055','162400','185944','414110','020615','849758','303394','168802','832801','440933','223978','605028','396087','369914','892511','241905','239039','843162','159905','278515','572988','049007','773358','929792','911142','172578','419312','556650','801985','363617','088470','703718','729115','114736','301423','472239','986005','483194','313380','706601','122365','117495','325615','421929','362264','667380','599051','310524','983546','369770','688939','255065','299159','902915','765255','336376','468891','383304','418863','275555','524340','284568','259021','566859','436441','572171','635627','222677','611405','883093','979516','754504','139750','070577','709587','929441','655974','991558','271056','447872','796783','055756','706079','181603','705063','052553','501997','114571','098990','769638','846404','717050','575406','865647','978861','080392','483124','734538','103119','082079','590362','018385','279399','050932','390355','846037','283923','192073','222714','142178','731302','848742','271015','028220','340449','785701','130313','510280','486684','809063','155536','096118','986549','647181','929022','061964','962355','005966','594456','624844','589315','200215','586774','028816','642356','079105','983906','804972','176679','201327','168572','017728','228690','330285','970877','359602','405692','994985','790458','417508','131300','423287','864460','971497','742696','626209','744819','943010','655943','516189','218573','778086','034458','972599','291901','341901','001242','383898','295123','664590','876883','032410','368396','909576','715809','713289','944575','183106','262505','905999','355773','577016','191049','890718','393864','365191','903951','505187','308188','233163','487579','921410','807111','245631','283552','168910','540142','678726','801113','535920','700107','942488','132433','761800','063697','629240','556533','182358','856387','688424','117446','568771','336162','003206','711573','815623','331633','561237','195181','265260','717413','005103','917620','619952','443029','352851','203301','572737','317648','021810','094730','713444','135453','689738','645657','518606','618718','126255','223744','578320','535420','976230','028050','839394','404607','509714','742816','092075','932905','858491','472209','191416','354237','934332','617685','668750','841593','553427','594292','711566','240803','560019','717294','532979','411932','511564','463678','569996','769694','327683','016744','942922','701205','555434','516090','218022','942135','262041','594088','030920','403839','873639','276033','609374','843882','388458','491842','853711','328421','818625','265810','415356','191954','365555','986532','619274','408954','860007','503516','767957','498434','670998','273421','370982','236853','071951','874277','548640','958965','126118','523699','641835','520461','632607','153029','805688','898663','658739','434801','711756','406863','441577','271784','595836','370865','975141','841893','767218','115010','497797','275642','166620','391493','066019','149873','990512','772033','881938','043986','770660','584505','651116','856648','742424','027777','636979','777878','525373','445669','766267','082262','882153','722885','015846','575957','325679','254093','544827','264574','466591','393901','250573','712730','070663','139808','769975','516392','919757','438063','018298','332370','258133','653462','299430','360576','148439','072815','486000','928711','494126','870454','751600','237333','174327','712438','066204','722922','283068','542343','516557','573224','308556','000023','972540','397422','593073','551756','642511','847490','783044','827036','347511','756776','168989','636013','417412','128824','015065','585441','066106','618142','752321','956759','201067','814617','958363','583122','778107','650534','644723','939685','288950','976006','051613','202635','658485','572690','186548','187317','312610','209145','943491','627166','027309','384666','327182','327663','257048','694463','242336','420500','464134','595675','172450','546798','087802','039149','648764','847623','337108','655857','463128','071951','457817','407497','135652','534501','742365','691694','943518','712379','731354','972904','457652','425077','954056','802454','655925','689662','835334','668701','429546','878870','526406','789369','422804','979167','989510','881270','541595','053233','097149','240522','323510','477803','001895','486197','957046','440494','848701','132546','862606','196612','656231','204454','319787','637466','127286','823622','266275','903202','479581','472676','151311','041455','136619','574862','116142','396221','682878','860705','081194','738478','368411','344557','070488','017798','332804','689896','520480','454653','481972','634529','136260','250349','405875','426292','027278','549130','465946','114722','465466','056906','729531','722473','189472','460276','060831','701624','900976','526436','577270','200477','667912','646027','481221','962568','389529','238906','403344','764549','939819','182773','266143','210459','964235','253225','734287','971497','367257','135704','144014','623715','347308','742015','137375','893023','411255','718183','162557','877297','443117','814799','577679','007003','712854','939641','717050','278934','292265','051720','658520','946483','363706','069207','157553','203054','200312','472442','657946','031204','533121','579064','321290','209894','467017','095814','688782','025632','498015','287375','936427','229422','419635','831614','797234','378641','939768','597459','835348','083470','049231','356943','958899','467566','004211','347066','009204','461910','718183','685324','312327','411798','857516','340803','956095','871573','646910','741514','206155','634174','217337','684886','167079','349570','568266','767640','859998','452384','665960','502483','914815','336132','429884','782406','675640','592710','273819','508080','928711','889465','927784','433840','177020','948731','853188','365686','112856','659614','590449','688193','079723','295260','261585','925921','651337','409220','217680','567214','055990','619205','862210','620792','192471','414556','875626','382920','897077','474948','567816','862280','259546','710567','490470','507813','605609','110470','367573','250007','493588','393888','754920','873543','090337','759239','863814','139808','382398','142759','273838','345709','240643','317783','688388','046892','399007','747731','862988','788969','050234','559119','479519','241957','677493','643706','456942','869149','020619','351884','861717','852991','720720','511534','566646','579666','062540','129183','486983','311672','300342','866381','953326','486633','142678','609899','654253','734455','659804','342753','159424','255065','577112','231262','369495','387810','764064','561237','570923','479917','819050','686360','893274','184861','912062','928505','379045','207769','923009','703427','808629','423736','032352','286344','100484','783509','168572','910130','377246','403675','188634','465253','741212','824716','604012','510981','553626','548221','846316','014345','012705','366756','298445','185577','605947','183243','770292','703912','211733','528939','664560','486216','451475','953539','558183','911810','845646','209427','046610','364340','723489','738952','772878','364505','164682','797881','176452','274293','778618','226620','068875','564842','981064','104242','870038','940918','933961','881560','985419','456856','168572','934785','029568','589652','980846','459185','051662','074465','935883','209611','631361','171510','019764','789319','916246','740053','425693','274004','825485','746944','111967','207807','374451','310243','981752','018903','535720','896363','249035','074153','046874','365686','292420','035536','023261','933326','384471','893604','277982','067025','769143','674556','596259','702394','385153','875492','446162','162081','751014','775132','932644','511681','961971','071882','742672','401138','095763','163229','967527','993566','224828','285669']
test_ans_list = ['244461','618355','700801','602278','891535','343118','311791','814953','716030','037451','813687','948577','461314','364921','184857','791725','881453','814274','032888','869517','280982','085584','496629','981851','705712','395291','558873','438042','588152','428139','457868','727311','019785','146772','634395','435391','606887','019338','155415','210230','792211','486197','952921','565380','908477','968495','264437','868048','306248','085601','091460','498228','336792','417508','097605','427896','536141','839861','745957','163844','814806','859075','746154','744191','436441','833814','184493','481486','170535','309824','897830','843133','298057','042348','798955','850955','802517','016297','135986','826841','272803','631075','576677','599779','321536','247970','143518','380593','012475','467854','424903','717596','815492','387171','981690','226049','088772','335431','780306','229492']

# 訓練圖檔目錄，目錄下僅能有圖檔
data_path = './images/amigodog_train2'
test_data_path = './images/amigodog_test'

# 讀檔
def img_loader(img_path):
    img = Image.open(img_path)
    return img.convert("RGB")
    
# 配對訓練圖與答案
def make_dataset(data_path,ans_list,ans_text,ans_text_length,ans_length):
    img_names = os.listdir(data_path)# 取出圖片名稱
    img_names.sort(key=lambda x: int(x.split(".")[0]))# 讓圖片從小到大排序
    samples = []
    
    # 用zip將圖片跟對應的答案湊一對
    for ans, img_name in zip(ans_list, img_names):
        if len(str(ans)) == ans_text_length:
            # 將圖片名稱及路徑合併
            # 以便上述程式img_loader的執行
            img_path = os.path.join(data_path, img_name)
            target = []
            for char in str(ans):
                vec = [0] * ans_length
                vec[ans_text.find(char)] = 1
                target += vec
                
            samples.append((img_path, target))
        else:
            print(img_name)
            
    return samples
    
class CaptchaData(Dataset):
    def __init__(self, data_path,ans_list,ans_text_length,ans_length,
                 transform=None, target_transform=None, ans_text=ans_text):
        super(Dataset, self).__init__()
        self.data_path = data_path
        self.ans_list = ans_list
        self.ans_text_length = ans_text_length
        self.ans_length = ans_length
        self.transform = transform
        self.target_transform = target_transform
        self.ans_text = ans_text
        self.samples = make_dataset(self.data_path,self.ans_list,self.ans_text,self.ans_text_length,self.ans_length
                                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, target = self.samples[index]
        img = img_loader(img_path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, torch.Tensor(target)
   
# AttentionBlock, ModifiedDenseNet201 注意力機制，額外處理文字位置不固定問題
class AttentionBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(AttentionBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1)
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1)

    def forward(self, x):
        out = self.pool(x)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = torch.sigmoid(out)
        out = x * out
        return out
        
class ModifiedDenseNet201(nn.Module):
    def __init__(self, num_classes, attention=False):
        super(ModifiedDenseNet201, self).__init__()
        self.densenet = models.densenet201(pretrained=True)
        self.attention = attention

        if self.attention:
            self.att_block1 = AttentionBlock(256)
            self.att_block2 = AttentionBlock(512)
            self.att_block3 = AttentionBlock(1792)

        self.fc = nn.Linear(1920, num_classes)  # 根据您的任务调整输入大小

    def forward(self, x):
        features = self.densenet.features(x)

        if self.attention:
            features = self.att_block1(features)
            features = self.att_block2(features)
            features = self.att_block3(features)

        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out
        
def calculat_acc(output, target, ans_length):
    output, target = output.view(-1, ans_length), target.view(-1, ans_length)
    output = nn.functional.softmax(output, dim=1)
    output = torch.argmax(output, dim=1)
    target = torch.argmax(target, dim=1)
    output, target = output.view(-1, 5), target.view(-1, 5)
    correct_list = []
    for i, j in zip(target, output):
        if torch.equal(i, j):
            correct_list.append(1)
        else:
            correct_list.append(0)
    acc = sum(correct_list) / len(correct_list)
    return acc
    
def train():
    transforms = Compose([ToTensor()])
    train_dataset = CaptchaData(data_path,
                                ans_list,
                                ans_text_length,
                                ans_length,
                                transform=transforms)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0,
                                   shuffle=True, drop_last=True)
    test_data = CaptchaData(test_data_path,
                            test_ans_list,
                            ans_text_length,
                            ans_length,
                            transform=transforms)
    test_data_loader = DataLoader(test_data, batch_size=batch_size,
                                  num_workers=0, shuffle=True, drop_last=True)
                                  
    # 使用densenet201來做訓練
    cnn = ModifiedDenseNet201(num_classes=ans_text_length * ans_length, attention=False) # 使用注意力機制(解決驗證碼位置不固定，準確率低問題) 
    # cnn = models.densenet201(num_classes=ans_text_length * ans_length) # 未使用注意力機制
    
    # 測試有沒有裝cuda，有沒有GPU可以使用
    if torch.cuda.is_available():
        cnn.cuda()
    # if restor:
        # cnn.load_state_dict(torch.load(model_path))
        
    # 優化器使用SGD+momentum，搭配CosineAnnealing（餘弦退火）的學習率scheduler，可以不固定學習率，以防他停在局部低點，有機會找到全局最佳解。
    optimizer = torch.optim.SGD(cnn.parameters(), lr=base_lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=16, eta_min=0, last_epoch=-1, verbose=False)
    criterion = nn.MultiLabelSoftMarginLoss()

    for epoch in range(max_epoch):
        start_ = time.time()

        loss_history = []
        acc_history = []
        cnn.train()

        for img, target in train_data_loader:
            # img = Variable(img)
            # target = Variable(target)
            if torch.cuda.is_available():
                img = img.cuda()
                target = target.cuda()
            output = cnn(img)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = calculat_acc(output, target, ans_length)
            acc_history.append(float(acc))
            loss_history.append(float(loss))
        scheduler.step()
        
        print(optimizer.param_groups[0]['lr'])
        print('train_loss: {:.4}|train_acc: {:.4}'.format(
            torch.mean(torch.Tensor(loss_history)),
            torch.mean(torch.Tensor(acc_history)),
        ))

        loss_history = []
        acc_history = []
        cnn.eval()
        for img, target in test_data_loader:
            # img = Variable(img)
            # target = Variable(target)
            if torch.cuda.is_available():
                img = img.cuda()
                target = target.cuda()
            output = cnn(img)

            acc = calculat_acc(output, target, ans_length)
            acc_history.append(float(acc))
            loss_history.append(float(loss))
            
        print('test_loss: {:.4}|test_acc: {:.4}'.format(
            torch.mean(torch.Tensor(loss_history)),
            torch.mean(torch.Tensor(acc_history)),
        ))
        print('epoch: {}|time: {:.4f}'.format(epoch, time.time() - start_))
        print("========================================")
        torch.save(cnn.state_dict(), model_path)

train()