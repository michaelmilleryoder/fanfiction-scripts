library('stm')

# Settings
do_stem <- TRUE
num_topics <- 10
max_iters <- 50
min_df <- 100 # minimum document frequency for words

cat('Loading data...')
dataset_name <- '2fandoms'
#fandom <- 'harrypotter'
#data <- read.csv(sprintf('/data/fanfiction_ao3/%s/complete_en_1k-50k/output/character_relationship_features.csv', fandom))
data <- read.csv(sprintf('/data/fanfiction_ao3/character_features_%s.csv', dataset_name))
covariates <- 'gender+rel_type'
#covariates <- 'gender'

cat('Processing data...')
processed <- textProcessor(data$character_features, stem=do_stem, metadata=data)
out <- prepDocuments(processed$documents, processed$vocab, processed$meta,
					lower.thresh=min_df)

cat('\nEstimating STM...')
estimated <- stm(documents = out$documents,
                vocab = out$vocab,
                K = num_topics,
                max.em.its = max_iters,
                prevalence =~ character_gender + character_in_relationship_type,
                #prevalence =~ character_gender,
                data = out$meta)

cat('Saving model...')
outpath <- sprintf('/projects/fanfiction_gender_roles/models/%s_stm_%s_%dtopics_%dit_%dmindf.rds', dataset_name, covariates, num_topics, max_iters, min_df)
saveRDS(estimated, outpath)
