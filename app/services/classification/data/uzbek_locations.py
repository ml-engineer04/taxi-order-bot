"""Uzbekistan geographical locations, cities, districts, and transit hubs."""

# Canonical city and regional hub names in normalized Latin
UZBEK_CITIES = {
    # Tashkent region & City
    "toshkent", "tashkent", "chirchiq", "olmaliq", "angren", "bekobod", "yangiyo'l", "yangiyol",
    "parkent", "piskent", "buka", "bo'ka", "chinoz", "qibray", "zangiota",
    "bo'stonliq", "bostonliq", "chig'atoy", "chigatoy", "qo'yliq", "qoyliq", "sergeli", "chilonzor", "yunusobod",
    "aeroport", "vokzal", "markaz",

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
}

# Suffixes indicating departure / origin ("from")
ORIGIN_SUFFIXES = ("dan", "dan", "dagi")

# Suffixes indicating arrival / destination ("to")
DESTINATION_SUFFIXES = ("ga", "ka", "qa", "gacha", "kacha", "qacha")
