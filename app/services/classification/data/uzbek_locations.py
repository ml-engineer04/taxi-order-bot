"""Uzbekistan geographical locations, cities, districts, and transit hubs."""

# Canonical city and regional hub names in normalized Latin
UZBEK_CITIES = {
    # Tashkent region & City
    "toshkent", "tashkent", "chirchiq", "olmaliq", "angren", "bekobod", "yangiyo'l", "yangiyol",
    "parkent", "piskent", "buka", "bo'ka", "chinoz", "qibray", "zangiota",
    "bo'stonliq", "bostonliq", "chig'atoy", "chigatoy", "qo'yliq", "qoyliq", "sergeli", "chilonzor", "yunusobod",
    "aeroport", "vokzal", "markaz", "shtab", "skver",

    # Samarkand
    "samarqand", "samarkand", "kattaqo'rg'on", "kattaqorgon", "urgut", "bulung'ur", "bulungur",
    "oqdaryo", "jomvoy", "pastdarg'om", "pastdargom", "narpay", "payariq", "paxtachi", "toyloq",

    # Bukhara & Navoiy
    "buxoro", "bukhara", "g'ijduvon", "gijduvon", "kogon", "qorako'l", "qorakol",
    "shofirkon", "vobkent", "jondor", "romitan", "peshku",
    "navoiy", "zarafshon", "uchquduq", "karmana", "qiziltepa", "xatirchi",

    # Fergana Valley
    "vodiy", "farg'ona", "fargona", "fergana", "qo'qon", "qoqon", "marg'ilon", "margilon",
    "quva", "quvasoy", "rishton", "oltiariq", "yaypan", "buvayda", "beshariq", "uchko'prik",
    "uchkoprik", "bog'dod", "bogdod", "dang'ara", "dangara", "furqat", "toshloq", "yazyovon", "yozyovon", "sox",
    "andijon", "andijan", "asaka", "shahrixon", "xo'jaobod", "xojaobod", "qorasuv", "marhamat",
    "paxtaobod", "baliqchi", "bo'ston", "boston", "buloqboshi", "izboskan", "jalaquduq", "qo'rg'ontepa", "qorgontepa",
    "namangan", "chust", "pop", "kosonsoy", "uchqo'rg'on", "uchqorgon", "yangiqo'rg'on", "yangiqorgon",
    "chortoq", "mingbuloq", "uychi", "norin",

    # Kashkadarya & Surkhandarya
    "qarshi", "karshi", "shahrisabz", "kitob", "yakkabog'", "yakkabog", "koson", "muborak", "g'uzor", "guzor",
    "chiroqchi", "qamashi", "nishon", "kasbi",
    "termiz", "denov", "sho'rchi", "shorchi", "qumqo'rg'on", "qumqorgon", "sherobod", "boysun", "jarqo'rg'on", "jarqorgon",
    "sariosiyo", "uzun", "oltinsoy", "muzrabot", "angor",

    # Jizzakh & Syrdarya
    "jizzax", "jizzakh", "do'stlik", "dostlik", "zaamin", "zomin", "paxtakor", "g'allaorol", "gallaorol", "baxmal",
    "guliston", "yangiyer", "shirin", "boyovut", "baxt", "sardoba", "sayxunobod", "oqoltin",

    # Khorezm & Karakalpakstan
    "urganch", "urgench", "xiva", "khiva", "xonqa", "shovot", "hazorasp", "gurlan", "bog'ot", "bogot", "yangiariq",
    "nukus", "qo'ng'irot", "qongirot", "beruniy", "to'rtko'l", "tortkol", "xo'jayli", "xojayli",
    "chimboy", "mang'it", "mangit", "taxtako'pir", "taxtakopir", "amudaryo",

    # Additional Tashkent districts, transit markets, and Fergana/regional hubs
    "chorsu", "toshmi", "olmazor", "alamazor", "yashnobod", "uchtepa", "yakkasaroy",
    "mirobod", "mirabad", "bektemir", "shayxontohur", "yangihayot", "qoraqamish",
    "farhod", "otchopar", "rohat", "roxat", "vodnik", "ttz", "fudsiti", "fud",
    "bachqir", "rapqon", "oqqo'rg'on", "oqqorgon", "okqorgon", "yangibozor", "navbahor",
    "chig'atoy", "chilanzor", "sergili", "sirgali", "sirg'ali", "koyluq", "quyluq",
    "qo'yluq", "kuylyuk", "qoqon", "quqon", "kokon", "toshken", "toshkon", "tosh", "samarqan",
    "besharik", "beshriq", "bagdod", "bagdo", "bogdo", "uchkopri", "uchkopirik", "fargon", "andijo", "namanga",
    "sadoqat", "sadokhat", "yakkatut", "tamasha", "erkin", "uchuy", "sho'rsuv", "shorsuv", "choshtepa",
    "qoplonbek", "shohimardon", "chashma", "xovos", "sirdaryo",
}

# Map colloquial/truncated/typo stems to canonical display form
CITY_ALIASES = {
    "tosh": "toshkent",
    "toshken": "toshkent",
    "toshkon": "toshkent",
    "samarqan": "samarqand",
    "quqon": "qo'qon",
    "kokon": "qo'qon",
    "qoqon": "qo'qon",
    "besharik": "beshariq",
    "beshriq": "beshariq",
    "sergili": "sergeli",
    "sirgali": "sergeli",
    "sirg'ali": "sergeli",
    "koyluq": "qo'yliq",
    "quyluq": "qo'yliq",
    "qo'yluq": "qo'yliq",
    "kuylyuk": "qo'yliq",
    "chilanzor": "chilonzor",
    "bagdod": "bog'dod",
    "bagdo": "bog'dod",
    "bogdo": "bog'dod",
    "uchkopri": "uchko'prik",
    "uchkopirik": "uchko'prik",
    "mirabad": "mirobod",
    "alamazor": "olmazor",
    "roxat": "rohat",
    "okqorgon": "oqqo'rg'on",
    "oqqorgon": "oqqo'rg'on",
    "fargon": "farg'ona",
    "andijo": "andijon",
    "namanga": "namangan",
    "sadokhat": "sadoqat",
    "shorsuv": "sho'rsuv",
    "parken": "parkent",
    "yazyavan": "yozyovon",
}

# Suffixes indicating departure / origin ("from")
ORIGIN_SUFFIXES = ("dan", "tan", "dagi")

# Suffixes indicating arrival / destination ("to")
DESTINATION_SUFFIXES = ("ga", "ka", "qa", "ge", "ke", "qe", "gacha", "kacha", "qacha", "geta", "keta", "qeta")
