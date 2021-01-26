# Twitter_topics

NB apriori algorithm works well also on Australia algorithm, while naive crashes

parlare del caso Russia -> row 24 (22) of covid apriori no sigletons

dato che le parole frequenti non sono stopwords, non ha senso eliminarle dai risultati
perché se in un certo periodo di tempo sono molto frequenti significa che sono significative
per quel periodo

NAIVE METHOD:
1) in naive top-k i top-k most common possono cambiare, dato che se ci sono elementi a pari merito solo il primo
fra essi è restituito, di conseguenza è un metodo non affidabile, dato che scarta elementi che
potrebbero essere invece da considerarsi frequenti, inoltre è difficile decidere un top-k che vada
bene per tutti i set di tweets, dato che ogni set ha una dimensione diversa, è quindi piu corretto
considerare la frequenza
2) quindi si è deciso di considerare nive freq, in cui la scelta dei topic frequenti è più coerente,
qui il problema è però dato dalla scalabilità, non riesce a computare su set di tweet grandi

APRIORI METHOD:
permette di risolvere il problema di scalabilità e di scegliere coerentemente i topic frequenti
   