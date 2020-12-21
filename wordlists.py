# These words are written as the stem to make it easier to match all variants.

feminine_coded_words = [
    u"angenehm",
    u"aufrichtig",
    u"beraten",
    u"bescheiden",
    u"betreu",
    u"beziehung",
    u"commit",
    u"dankbar",
    u"ehrlich",
    u"einfühl",
    u"emotion",
    u"empath",
    u"engag",
    u"familie",
    u"fleiß",
    u"förder",
    u"freundlich",
    u"freundschaft",
    u"fürsorg",
    u"gefühl",
    u"gemeinsam",
    u"gemeinschaft",
    u"gruppe",
    u"harmon",
    u"helfen",
    u"herzlich",
    u"hilf",
    u"höflich",
    u"interpers",
    u"kollabor",
    u"kollegial",
    u"kooper",
    u"kümmern",
    u"liebensw",
    u"loyal",
    u"miteinander",
    u"mitfühl",
    u"mitgefühl",
    u"mithelf",
    u"mithilf",
    u"nett",
    u"partnerschaftlich",
    u"pflege",
    u"rücksicht",
    u"sensib",
    u"sozial",
    u"team",
    u"treu",
    u"umgänglich",
    u"umsichtig",
    u"uneigennützig",
    u"unterstütz",
    u"verantwortung",
    u"verbunden",
    u"verein",
    u"verlässlich",
    u"verständnis",
    u"vertrauen",
    u"wertschätz",
    u"zugehörig",
    u"zusammen",
    u"zuverlässig",
    u"zwischenmensch"
]

masculine_coded_words = [
    u"abenteuer",
    u"aggressiv",
    u"ambition",
    u"analytisch",
    u"aufgabenorientier",
    u"autark",
    u"autoritä",
    u"autonom",
    u"beharr",
    u"besieg",
    u"sieg",
    u"bestimmt",
    u"direkt",
    u"domin",
    u"durchsetz",
    u"ehrgeiz",
    u"eigenständig",
    u"einzel",
    u"einfluss",
    u"einflussreich",
    u"energisch",
    u"entscheid",
    u"entschlossen",
    u"erfolgsorientier",
    u"führ",
    u"anführ",
    u"gewinn",
    u"hartnäckig",
    u"herausfordern",
    u"hierarch",
    u"kompetitiv",
    u"konkurr",
    u"kräftig",
    u"kraft",
    u"leisten",
    u"leistungsfähig",
    u"leistungsorient",
    u"leit",
    u"anleit",
    u"lenken",
    u"mutig",
    u"offensiv",
    u"persisten",
    u"rational",
    u"risiko",
    u"selbstbewusst",
    u"selbstsicher",
    u"selbstständig",
    u"selbständig",
    u"selbstvertrauen",
    u"stark",
    u"stärke",
    u"stolz",
    u"überlegen",
    u"unabhängig",
    u"wettbewerb",
    u"wetteifer",
    u"wettkampf",
    u"wettstreit",
    u"willens",
    u"zielorient",
    u"zielsicher",
    u"zielstrebig"
]

hyphenated_coded_words = [
    #"co-operat",
    #"inter-personal",
    #"inter-dependen",
    #"inter-persona",
    #"self-confiden",
    #"self-relian",
    #"self-sufficien"
]

possible_codings = (
    "überwiegend agentisch (stereotyp männlich) formuliert",
    "überwiegend kommunal (stereotyp weiblich) formuliert",
    "neutral formuliert",
)

explanations = {
    "überwiegend kommunal (stereotyp weiblich) formuliert": (
        "Nach unserer Auswertung enthält Ihre Stellenanzeige mehr kommunale "
        "(stereotyp weibliche) als agentische (stereotyp männliche) Worte. "
        "Ihre Stellenanzeige könnte daher Frauen darin bestätigen, sich zu "
        "bewerben. Forschungsergebnisse zeigen, dass eine kommunale "
        "Formulierung einer Stellenanzeige kaum Effekte darauf hat, wie attraktiv "
        "Männer eine Stelle wahrnehmen, sodass die kommunale "
        "Formulierung Ihrer Stellenanzeige Männer vermutlich nicht davon abhalten "
        "wird, sich zu bewerben."),
    "überwiegend agentisch (stereotyp männlich) formuliert": (
        "Nach unserer Auswertung enthält Ihre Stellenanzeige mehr "
        "agentische (stereotyp männliche) als kommunale "
        "(stereotyp weibliche) Worte. Empirischen Studien zufolge "
        "spricht diese Wortwahl eher Männer als Frauen an und kann "
        "Frauen davon abhalten, sich zu bewerben. Durch eine agentische "
        "Formulierung Ihrer Stellenanzeige kann der Eindruck "
        "entstehen, dass die Anforderungen der Stelle vorwiegend agentische "
        "Eigenschaften beinhalten und auch eher Personen, die "
        "diesem Profil entsprechen, gesucht werden."),
    "stark feminin (kommunal) konnotiert": (
        "Nach unserer Auswertung enthält Ihre Stellenanzeige mehr stereotyp "
        "weibliche (kommunale) als stereotyp männliche (agentische) Worte. "
        "Ihre Stellenanzeige könnte daher Frauen darin bestätigen, sich zu "
        "bewerben. Forschungsergebnisse zeigen, dass eine stereotyp weibliche "
        "Formulierung einer Stellenanzeige kaum Effekte darauf hat, wie attraktiv "
        "Männer eine Stelle wahrnehmen, sodass die stereotyp weibliche "
        "Formulierung Ihrer Stellenanzeige Männer vermutlich nicht davon abhalten "
        "wird, sich zu bewerben."),
    "stark maskulin (agentisch) konnotiert": (
        "Nach unserer Auswertung enthält Ihre Stellenanzeige mehr "
        "stereotyp männliche (agentische) als stereotyp weibliche "
        "(kommunale) Worte. Empirischen Studien zufolge "
        "spricht diese Wortwahl eher Männer als Frauen an und kann "
        "Frauen davon abhalten, sich zu bewerben. Durch eine stereotyp "
        "männliche Formulierung Ihrer Stellenanzeige kann der Eindruck "
        "entstehen, dass die Anforderungen der Stelle vorwiegend stereotyp "
        "männliche Eigenschaften beinhalten und auch eher Personen, die "
        "diesem Profil entsprechen, gesucht werden."),
    "empty": (
        "Nach unserer Auswertung enthält Ihre Stellenanzeige "
        "weder agentische (stereotyp männliche) noch kommunale "
        "(stereotyp weibliche) Worte. Folglich wird durch Ihre "
        "Stellenanzeige weder suggeriert, dass vorrangig agentische "
        "Eigenschaften gesucht und gefordert werden, noch dass "
        "vorrangig kommunale Eigenschaften gesucht und gefordert "
        "werden. Die sprachliche Gestaltung Ihrer Stellenanzeige "
        "ist demnach gender-fair."),
    "neutral formuliert": (
        "Nach unserer Auswertung enthält Ihre Stellenanzeige "
        "gleichermaßen agentische (stereotyp männliche) und "
        "kommunale (stereotyp weibliche) Worte. Folglich wird durch Ihre "
        "Stellenanzeige weder suggeriert, dass vorrangig agentische "
        "Eigenschaften gesucht und gefordert werden, noch dass "
        "vorrangig kommunale Eigenschaften gesucht und gefordert "
        "werden. Die sprachliche Gestaltung Ihrer Stellenanzeige "
        "ist demnach gender-fair."
        )
}


salary_words = [
    "eur",
    "euro",
]

profit_words = [
    "umsatz", 
    "jahresumsatz", 
    "budget"
]

workingtime_words = [
    "teilzeit",
    "arbeitszeit",
    "jobsharing",
    "arbeitsplatzzeitteilung",
    "arbeitszeitmodell",
    "belastbarkeit"
]

family_words = [
    "kita",
    "kinderbetreuung",
    "familiär",
    "kinder",
    "familienfreundlich",
    "elternkarenz",
    "familienservice"
]

homeoffice_words = [
    "homeoffice",
    "home-office",
    "remote"
]


safety_words = [
    "unbefristet",
    "altersvorsorge",
    "festanstellung",
    "permanent"
]


health_words = [
    "betriebsarzt",
    "gesundheitsförderung",
    "gesundheitstag",
    "gesundheitscheck",
]

travel_words = [
    "international",
    "reisetätigkeit",
    "reisebereitschaft",
    "dienstreise"
]