import random
import time

#définition des colonnes pour l'extraction des données
#peut-être modifié selon les informations qui nous intéressent
WORD_INDEX = 1
LABEL_INDEX = 3

#définition des variables globales
WORD = "mot"
GOLD = "gold"
PREV_WORD = "prev_word"
NEXT_WORD = "next_word"
BEGIN_TOKEN = "XXXBEGINXXX"
END_TOKEN = "XXXENDXXX"

NB_FEATURES = 9
#Liste des noms qui permettent d'initialiser tous les poids
LABELS = ['DET', 'ADJ', '_', 'NUM', 'VERB', 'ADP', 'PART', 'CCONJ', 'X', 'INTJ', 'ADV', 'NOUN', 'PROPN', 'PRON', 'SYM', 'SCONJ', 'AUX', 'PUNCT']

#Listes envisagées pour certaines caractéristiques mais abandonnées car inefficaces voire contre-productives

#LISTE_DET = ["le", "la", "les", "l'", "un", "une", "des", "mon", "ma", "mes", "ton", "ta", "tes", "son", "sa", "ses", "notre", "nos", "votre", "vos" , "leur", "leurs", "ce", "cette", "cet", "ces"]
#LISTE_PRON = ["je", " me", " m’", " moi ", " tu", " te", " t’", " toi", " nous", " vous ", " il", " elle", " ils", " elles ", " se"]
#LISTE_PONCTUATIONS = [".", ",", ":", "?", ";"]
#features_dict = init_features()

def extract_words_from_file(filename):
	corp = ""
	with open(filename,'r') as fi:
		for line in fi:
			if not line.startswith("#"):
				corp += line
	#print(corp)
	word_entries = []
	for sentence in corp.split("\n\n") :
		if len(sentence) >= 3 :
			temp_word_entries = []
			words_in_sentence = sentence.split("\n")
			prev_word = BEGIN_TOKEN
			current_word_info = words_in_sentence[0].split("\t")
			for i in range(len(words_in_sentence) - 1):
				next_word_info = words_in_sentence[i + 1].split("\t")
				word_entries.append(create_word_entry(current_word_info, prev_word, next_word_info[WORD_INDEX]))
				prev_word = current_word_info[WORD_INDEX]
				current_word_info = next_word_info
			word_entries.append(create_word_entry(current_word_info, prev_word, END_TOKEN))
	return word_entries

# word = liste caractéristiques mot dans phrase
# prev_word = string mot (ex: "table")
# next_word = string mot
#sortie type: {"mot": "chat", "prev_word": "le", "next_word": "mange", "gold": "NOUN"}
def create_word_entry(word, prev_word, next_word):
	entry = {}
	entry[WORD] = word[WORD_INDEX]
	entry[PREV_WORD] = prev_word
	entry[NEXT_WORD] = next_word
	entry[GOLD] = word[LABEL_INDEX]
	return entry

def word_to_vect(word_entry, train):
	vect = {}
	MAJ_FEATURE = "maj"
	LONG_FEATURE = "long"
	SHORT_FEATURE = "short"


	vect[(PREV_WORD, word_entry[PREV_WORD])] = 1
	vect[(NEXT_WORD, word_entry[NEXT_WORD])] = 1

	if (word_entry[WORD][0].isupper()):
		vect[MAJ_FEATURE] = 1

	if len(word_entry[WORD]) < 4:
		vect[SHORT_FEATURE] = 1

	if len(word_entry[WORD]) > 6:
		vect[LONG_FEATURE] = 1
    #features abandonnées car inefficaces voire contre-productives
	"""if train:
		vect[("label", word_entry[GOLD])] = 1
	if (word_entry[PREV_WORD].lower() in LISTE_DET):
		f_mot_apres_det = 1

	if (word_entry[PREV_WORD].lower() in LISTE_PRON):
		f_mot_apres_pron = 1

	if word_entry[WORD] in LISTE_PONCTUATIONS:
		f_ponctuation = 1

	if word_entry[WORD].isdigit():
		f_nb = 1"""



	return vect

def init_weights():
	weights = {}
	for label in LABELS:
		weights[label] = {}
	return weights

def predict(word_vect, weights):
	labels_scores = {}
	for label, label_weights in weights.items():
		labels_scores[label] = sum(word_vect.get(feat) * label_weights.get(feat, 0) for feat in word_vect)
	return max(labels_scores, key=labels_scores.get)


def train_perceptron(training_set, MAX_EPOCH):
	a = init_weights()
	w = init_weights()
	for i in range(MAX_EPOCH):
		random.shuffle(training_set)
		last_update = {}
		n_examples = 0
		for word_entry in training_set:
			gold = word_entry[GOLD]
			word_vect = word_to_vect(word_entry, True)
			pred = predict(word_vect, w)
			if not gold == pred:
				for feat in word_vect:
					a[gold][feat] = a[gold].get(feat, 0)
					a[gold][feat] += (n_examples - last_update.get((gold, feat), 1)) * w[gold].get(feat, 1)
					last_update[(gold, feat)] = n_examples

					a[pred][feat] = a[pred].get(feat, 0)
					a[pred][feat] += (n_examples - last_update.get((pred, feat), 1)) * w[pred].get(feat, 1)
					last_update[(pred, feat)] = n_examples

					w[gold][feat] = w[gold].get(feat, 0) + 1
					w[pred][feat] = w[pred].get(feat, 0) - 1
			n_examples += 1
		for label, features in a.items():
			for feat, val in features.items():
				a[label][feat] += (n_examples - last_update.get((label, feat), 1)) * w[label].get(feat, 1)
	#print_a(a)
	return a

def evaluate(test_set, weights_vector, MAX_EPOCH):
    errors_count = 0
    confusion_matrix = {}
    for gold in LABELS:
        confusion_matrix[gold] = {}
        for pred in LABELS:
            confusion_matrix[gold][pred] = 0

    for word in test_set:
        pred = predict(word_to_vect(word, False), weights_vector)
        if not pred == word[GOLD]:
            confusion_matrix[word[GOLD]][pred] += 1
            errors_count += 1
    print_confusion_matrix(confusion_matrix)
    return errors_count / len(test_set) * 100

def print_a(a):
	for label in a.keys():
		print(label + ": ")
		c = 0
		for feat in a[label].items():
			if (c == 20):
				break
			print(feat)
			c += 1
def print_confusion_matrix(matrice_confusion):
    resultat = "      |"
    for e in matrice_confusion["NOUN"]:
        resultat += "{:<6}".format(e)
    resultat += "\n--------------------------------------------------------------------------------------------------------------------\n"
    for ref in matrice_confusion:
        resultat += "{:<6}|".format(ref)
        for pred in matrice_confusion[ref]:
            resultat += "{:<6}".format(matrice_confusion[ref][pred])
        resultat += "\n"
    print(resultat)

def run_test(word_entries_train, word_entries_test, MAX_EPOCH):
	weights = init_weights()
	weights_vector = train_perceptron(word_entries_train, MAX_EPOCH)
	return evaluate(word_entries_test, weights_vector, MAX_EPOCH)

train_filename = "fr_gsd-ud-train.conllu.txt"
test_filename = "fr_gsd-ud-test.conllu.txt"
word_entries_train = extract_words_from_file(train_filename) #---> training set
word_entries_test = extract_words_from_file(test_filename) #---> testing set

#évaluations pour trouver la valeur de MAX_EPOCH qui minimise le taux d'erreur
for i in range(1, 11):
	begin = time.time()
	err_rate = run_test(word_entries_train,word_entries_test, i)
	print ("Taux erreur: {:.4f}% ({} epochs)".format(err_rate, i) + " - Temps d'execution: {}".format(time.time() - begin))
