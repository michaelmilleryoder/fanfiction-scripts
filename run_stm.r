library('stm')

print('Loading data...')
#data <- read.csv('/data/fanfiction_gender_roles/harrypotter_ao3/complete_en_1k-50k/output/character_gender_features.csv')
data <- read.csv('/data/fanfiction_gender_roles/harrypotter_ao3/complete_en_1k-50k/output/character_relationship_features.csv')
#covariates <- 'gender+rel_type'
covariates <- 'gender'

# Settings
do_stem <- TRUE
num_topics <- 10
max_iters <- 50
min_df <- 100 # minimum document frequency for words

print('Processing data...')
processed <- textProcessor(data$character_features, stem=do_stem, metadata=data)
out <- prepDocuments(processed$documents, processed$vocab, processed$meta,
					lower.thresh=min_df)

print('\nEstimating STM...')
estimated <- stm(documents = out$documents,
                vocab = out$vocab,
                K = num_topics,
                max.em.its = max_iters,
                #prevalence =~ character_gender + character_in_relationship_type,
                prevalence =~ character_gender,
                data = out$meta)

print('Saving model...')
outpath <- sprintf('/projects/fanfiction_gender_roles/models/hp_stm_%s_%dtopics_%dit_%dmindf.rds', covariates, num_topics, max_iters, min_df)
saveRDS(estimated, outpath)
