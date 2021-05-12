library('stm')

# Settings
num_topics <- 10
max_iters <- 50
do_stem <- TRUE
min_df <- 50 # minimum document frequency for words for quotes
covariates <- 'dayxsexualityxdataset'

cat('Loading data...\n')
data <- read.csv('/data/fanfiction_lgbtq/charfeats_20fandoms.csv')
data$relationship_type <- as.factor(data$relationship_type)
data$dataset <- as.factor(data$dataset)
data$published <- as.Date(data$published)
data$day <- julian(data$published, origin=as.Date('2010-01-01'))

cat('Processing data...\n')
processed <- textProcessor(data$quote_features, stem=do_stem, metadata=data,
		verbose=TRUE)
out <- prepDocuments(processed$documents, processed$vocab, processed$meta,
					lower.thresh=min_df)

cat('\nEstimating STM...\n')
estimated <- stm(documents = out$documents,
                vocab = out$vocab,
                K = num_topics,
                max.em.its = max_iters,
                prevalence =~ s(day) * relationship_type * dataset,
				verbose=TRUE,
                data = out$meta)

outpath <- sprintf('/projects/fanfiction_lgbtq/models/quotes_stm_%s_%dtopics_%dit_%dmindf.rds', covariates, num_topics, max_iters, min_df)
cat(sprintf('Saved model to %s\n', outpath))
saveRDS(estimated, outpath)
