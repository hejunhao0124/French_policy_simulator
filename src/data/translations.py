"""
法国政策模拟器 - 法文→中文翻译映射

独立模块，避免 app.py 臃肿。
"""

import unicodedata
import pandas as pd


OCCUPATION_ZH = {
    "Ingénieur logiciel": "软件工程师", "Développeur web": "网站开发",
    "Enseignant": "教师", "Professeur": "教授", "Médecin": "医生",
    "Infirmier": "护士", "Comptable": "会计", "Avocat": "律师",
    "Architecte": "建筑师", "Commercial": "销售", "Consultant": "顾问",
    "Chef de projet": "项目经理", "Directeur": "总监", "Manager": "经理",
    "Responsable marketing": "市场主管", "Analyste financier": "金融分析师",
    "Data scientist": "数据科学家", "Graphiste": "平面设计师",
    "Journaliste": "记者", "Photographe": "摄影师",
    "Cuisinier": "厨师", "Boulanger": "面包师", "Serveur": "服务员",
    "Caissier": "收银员", "Vendeur": "售货员", "Électricien": "电工",
    "Plombier": "水管工", "Menuisier": "木匠", "Mécanicien": "机械师",
    "Conducteur": "司机", "Policier": "警察", "Pompier": "消防员",
    "Agent de sécurité": "保安", "Retraité": "退休", "Étudiant": "学生",
    "Au foyer": "家庭主妇/主夫", "Chômeur": "失业",
    "Agriculteur": "农民", "Ouvrier": "工人", "Employé de bureau": "文员",
    "Secrétaire": "秘书", "Assistante": "助理", "Technicien": "技术员",
    "Ingénieur": "工程师", "Pharmacien": "药剂师", "Dentiste": "牙医",
    "Psychologue": "心理学家", "Kinésithérapeute": "理疗师",
    "Avocat d'affaires": "商务律师", "Notaire": "公证人",
    "Artisan": "手工艺人", "Entrepreneur": "企业家",
    "Freelance": "自由职业者", "Fonctionnaire": "公务员",
    "Militaire": "军人", "Pilote": "飞行员",
    "Directeur financier": "财务总监", "Directeur commercial": "销售总监",
    "Directeur technique": "技术总监", "Directeur des ressources humaines": "人力资源总监",
    "Responsable logistique": "物流主管", "Responsable qualité": "质量主管",
    "Chercheur": "研究员", "Biologiste": "生物学家",
    "Chimiste": "化学家", "Mathématicien": "数学家",
    "Statisticien": "统计学家", "Économiste": "经济学家",
    "Sociologue": "社会学家", "Historien": "历史学家",
    "Écrivain": "作家", "Musicien": "音乐家",
    "Acteur": "演员", "Réalisateur": "导演",
    "Producteur": "制作人", "Animateur": "主持人",
    "Agent immobilier": "房产经纪", "Banquier": "银行家",
    "Assureur": "保险经纪", "Courtier": "经纪人",
    "Juriste": "法务", "Traducteur": "翻译",
    "Interprète": "口译", "Professeur de lycée": "高中教师",
    "Professeur des écoles": "小学教师", "Professeur d'université": "大学教授",
    "Éducateur": "教育工作者", "Animateur socio-culturel": "社会文化工作者",
    "Aide-soignant": "护工", "Sage-femme": "助产士",
    "Vétérinaire": "兽医", "Opticien": "眼镜师",
    "Coiffeur": "理发师", "Esthéticienne": "美容师",
    "Agent d'entretien": "保洁", "Garde d'enfants": "保姆",
    "Livreur": "快递员", "Agent de voyage": "旅行社职员",
    "Guide touristique": "导游", "Hôtelier": "酒店经理",
    "Réceptionniste": "前台", "Gérant de restaurant": "餐厅经理",
    "Sommelier": "侍酒师", "Pâtissier": "糕点师",
    "Boucher": "屠夫", "Fromager": "奶酪师",
    "Viticulteur": "葡萄种植者", "Pêcheur": "渔民",
    "Sylviculteur": "林业工人", "Jardinier": "园丁",
    "Paysagiste": "景观设计师", "Géomètre": "测量师",
    "Urbaniste": "城市规划师",
}

DEPARTEMENT_ZH = {
    "Ain": "安省(01)", "Aisne": "埃纳省(02)", "Allier": "阿列省(03)",
    "Alpes-de-Haute-Provence": "上普罗旺斯阿尔卑斯(04)",
    "Hautes-Alpes": "上阿尔卑斯省(05)", "Alpes-Maritimes": "滨海阿尔卑斯省(06)",
    "Ardèche": "阿尔代什省(07)", "Ardennes": "阿登省(08)",
    "Ariège": "阿列日省(09)", "Aube": "奥布省(10)",
    "Aude": "奥德省(11)", "Aveyron": "阿韦龙省(12)",
    "Bouches-du-Rhône": "罗讷河口省(13)", "Calvados": "卡尔瓦多斯省(14)",
    "Cantal": "康塔尔省(15)", "Charente": "夏朗德省(16)",
    "Charente-Maritime": "滨海夏朗德省(17)", "Cher": "谢尔省(18)",
    "Corrèze": "科雷兹省(19)", "Corse-du-Sud": "南科西嘉省(2A)",
    "Haute-Corse": "上科西嘉省(2B)", "Côte-d'Or": "科多尔省(21)",
    "Côtes-d'Armor": "阿摩尔滨海省(22)", "Creuse": "克勒兹省(23)",
    "Dordogne": "多尔多涅省(24)", "Doubs": "杜省(25)",
    "Drôme": "德龙省(26)", "Eure": "厄尔省(27)",
    "Eure-et-Loir": "厄尔-卢瓦尔省(28)", "Finistère": "菲尼斯泰尔省(29)",
    "Gard": "加尔省(30)", "Haute-Garonne": "上加龙省(31)",
    "Gers": "热尔省(32)", "Gironde": "吉伦特省(33)",
    "Hérault": "埃罗省(34)", "Ille-et-Vilaine": "伊勒-维莱讷省(35)",
    "Indre": "安德尔省(36)", "Indre-et-Loire": "安德尔-卢瓦尔省(37)",
    "Isère": "伊泽尔省(38)", "Jura": "汝拉省(39)",
    "Landes": "朗德省(40)", "Loir-et-Cher": "卢瓦-谢尔省(41)",
    "Loire": "卢瓦尔省(42)", "Haute-Loire": "上卢瓦尔省(43)",
    "Loire-Atlantique": "大西洋卢瓦尔省(44)", "Loiret": "卢瓦雷省(45)",
    "Lot": "洛特省(46)", "Lot-et-Garonne": "洛特-加龙省(47)",
    "Lozère": "洛泽尔省(48)", "Maine-et-Loire": "曼恩-卢瓦尔省(49)",
    "Manche": "芒什省(50)", "Marne": "马恩省(51)",
    "Haute-Marne": "上马恩省(52)", "Mayenne": "马耶讷省(53)",
    "Meurthe-et-Moselle": "默尔特-摩泽尔省(54)", "Meuse": "默兹省(55)",
    "Morbihan": "莫尔比昂省(56)", "Moselle": "摩泽尔省(57)",
    "Nièvre": "涅夫勒省(58)", "Nord": "北部省(59)",
    "Oise": "瓦兹省(60)", "Orne": "奥恩省(61)",
    "Pas-de-Calais": "加来海峡省(62)", "Puy-de-Dôme": "多姆山省(63)",
    "Pyrénées-Atlantiques": "大西洋比利牛斯省(64)",
    "Hautes-Pyrénées": "上比利牛斯省(65)",
    "Pyrénées-Orientales": "东比利牛斯省(66)",
    "Bas-Rhin": "下莱茵省(67)", "Haut-Rhin": "上莱茵省(68)",
    "Rhône": "罗讷省(69)", "Haute-Saône": "上索恩省(70)",
    "Saône-et-Loire": "索恩-卢瓦尔省(71)", "Sarthe": "萨尔特省(72)",
    "Savoie": "萨瓦省(73)", "Haute-Savoie": "上萨瓦省(74)",
    "Paris": "巴黎(75)", "Seine-Maritime": "滨海塞纳省(76)",
    "Seine-et-Marne": "塞纳-马恩省(77)", "Yvelines": "伊夫林省(78)",
    "Deux-Sèvres": "德塞夫勒省(79)", "Somme": "索姆省(80)",
    "Tarn": "塔恩省(81)", "Tarn-et-Garonne": "塔恩-加龙省(82)",
    "Var": "瓦尔省(83)", "Vaucluse": "沃克吕兹省(84)",
    "Vendée": "旺代省(85)", "Vienne": "维埃纳省(86)",
    "Haute-Vienne": "上维埃纳省(87)", "Vosges": "孚日省(88)",
    "Yonne": "约讷省(89)", "Territoire de Belfort": "贝尔福地区省(90)",
    "Essonne": "埃松省(91)", "Hauts-de-Seine": "上塞纳省(92)",
    "Seine-Saint-Denis": "塞纳-圣但尼省(93)",
    "Val-de-Marne": "马恩河谷省(94)", "Val-d'Oise": "瓦兹河谷省(95)",
    "Guadeloupe": "瓜德罗普(971)", "Martinique": "马提尼克(972)",
    "Guyane": "法属圭亚那(973)", "La Réunion": "留尼汪(974)",
}

EDUCATION_ZH = {
    "Bac+5 ou plus": "硕士及以上(Bac+5+)", "Bac+5": "硕士/研二(Bac+5)",
    "Bac+3 ou Bac+4": "本科至研一(Bac+3~4)", "Bac+3": "学士/大三(Bac+3)", "Bac+4": "研一(Bac+4)",
    "Bac+2": "大二(Bac+2)", "Bac+1": "大一(Bac+1)",
    "Baccalauréat": "高中毕业(Bac)", "CAP ou BEP": "职业证书(CAP/BEP)",
    "Sans diplôme ou CEP": "无学历", "Brevet": "初中毕业",
    "Doctorat": "博士", "Master": "硕士", "Licence": "学士",
    "Post-doctorat": "博士后", "Primaire": "小学", "Secondaire": "中学",
}

MARITAL_ZH = {
    "Célibataire": "单身", "Marié(e)": "已婚", "Pacsé(e)": "民事结合",
    "Divorcé(e)": "离婚", "Veuf/Veuve": "丧偶", "En couple": "同居",
    "Séparé(e)": "分居",
}

PCS_ZH = {
    "Agriculteurs exploitants": "农业从业者",
    "Artisans, commerçants, chefs d'entreprise": "工商业主/企业主",
    "Cadres et professions intellectuelles supérieures": "高管/高级知识分子",
    "Professions intermédiaires": "中级职业",
    "Employés": "雇员/职员",
    "Ouvriers": "工人",
    "Retraités": "退休人员",
    "Autres sans activité professionnelle": "其他无业人员",
}

HOUSEHOLD_ZH = {
    "Personne seule": "独居", "Couple sans enfant": "无子女夫妻",
    "Couple avec enfant(s)": "有子女夫妻", "Famille monoparentale": "单亲家庭",
    "Colocation": "合租", "Autre": "其他",
}

SEX_ZH = {"M": "男", "F": "女", "Homme": "男", "Femme": "女"}


def _normalize_fr(text):
    """去除法语重音，用于模糊匹配"""
    n = unicodedata.normalize('NFD', text)
    n = ''.join(c for c in n if unicodedata.category(c) != 'Mn')
    return n.strip().lower()


# 无重音版本映射（处理变音缺失）
OCCUPATION_NORM = {_normalize_fr(k): v for k, v in OCCUPATION_ZH.items()}
PCS_NORM = {_normalize_fr(k): v for k, v in PCS_ZH.items()}
EDUCATION_NORM = {_normalize_fr(k): v for k, v in EDUCATION_ZH.items()}
DEPARTEMENT_NORM = {_normalize_fr(k): v for k, v in DEPARTEMENT_ZH.items()}
MARITAL_NORM = {_normalize_fr(k): v for k, v in MARITAL_ZH.items()}
HOUSEHOLD_NORM = {_normalize_fr(k): v for k, v in HOUSEHOLD_ZH.items()}


def zh_translate(value, field_type="occupation"):
    """将法文字段翻译为中文（支持精确和模糊匹配）"""
    if pd.isna(value) or not value:
        return ""
    val = str(value).strip()
    if not val:
        return ""

    if field_type == "occupation":
        return (OCCUPATION_ZH.get(val)
                or PCS_ZH.get(val)
                or OCCUPATION_NORM.get(_normalize_fr(val))
                or val)
    elif field_type == "departement":
        return DEPARTEMENT_ZH.get(val) or DEPARTEMENT_NORM.get(_normalize_fr(val), val)
    elif field_type == "education_level":
        return EDUCATION_ZH.get(val) or EDUCATION_NORM.get(_normalize_fr(val), val)
    elif field_type == "marital_status":
        return MARITAL_ZH.get(val) or MARITAL_NORM.get(_normalize_fr(val), val)
    elif field_type == "household_type":
        return HOUSEHOLD_ZH.get(val) or HOUSEHOLD_NORM.get(_normalize_fr(val), val)
    elif field_type == "sex":
        return SEX_ZH.get(val, val)
    return val
