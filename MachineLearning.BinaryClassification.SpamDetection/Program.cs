using System;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Text;

namespace MachineLearning.BinaryClassification.SpamDetection
{
    public static class Program
    {
        public static void Main(string[] args)
        {
            // Define paths to files.
            var files =
                new[]
                {
                    @"Data/youtube_01_psy.csv",
                    @"Data/youtube_02_katy_perry.csv",
                    @"Data/youtube_03_lmfao.csv",
                    @"Data/youtube_04_eminem.csv",
                    @"Data/youtube_05_shakira.csv",
                };

            // Create ML context with defined seed.
            var mlContext = new MLContext(seed: 0);

            // Create text loader for loading multiple files.
            var textLoader =
                mlContext.Data.CreateTextLoader<Comment>(
                    separatorChar: ',',
                    hasHeader: true,
                    allowQuoting: true,
                    trimWhitespace: true);

            // Load all 5 files into IDataView.
            var data =
                textLoader.Load(files);

            // Define pipeline.
            var pipeline =
                new EstimatorChain<ITransformer>()
                    .Append(
                        mlContext.Transforms.Text.FeaturizeText(
                            "Features",
                            new TextFeaturizingEstimator.Options
                            {
                                CaseMode = TextNormalizingEstimator.CaseMode.Lower,
                                KeepDiacritics = false,
                                KeepNumbers = true,
                                KeepPunctuations = true,
                                StopWordsRemoverOptions = null, // Do not remove stopwords, we cannot be sure about language of comment.
                                CharFeatureExtractor = new WordBagEstimator.Options // Set custom parameters of features extranction from chars.
                                {
                                    NgramLength = 3,
                                    UseAllLengths = false,
                                    SkipLength = 0,
                                    MaximumNgramsCount = null,
                                    Weighting = NgramExtractingEstimator.WeightingCriteria.TfIdf,
                                },
                                WordFeatureExtractor = null, // Turn off word features extraction
                                Norm = TextFeaturizingEstimator.NormFunction.L2, // Normalize feature vector using L2 normaliation: https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.transforms.lpnormnormalizingestimator?view=ml-dotnet
                                OutputTokensColumnName = null, // Turn off generation of extra column.
                            },
                            nameof(Comment.Content))
                    .Append(
                        mlContext.Transforms.CopyColumns("Label", nameof(Comment.Class)))
                    .Append(
                        mlContext.BinaryClassification.Trainers.SdcaLogisticRegression()));

            // Perform cross validation
            var crossValidationResults =
                mlContext.BinaryClassification.CrossValidate(
                    data,
                    pipeline,
                    numberOfFolds: 5);

            // Print confusion matrix for each of cross validations.
            foreach (var result in crossValidationResults)
            {
                Console.WriteLine($"Fold {result.Fold}, accuracy: {result.Metrics.Accuracy}");
            }

            // Calculate mean accuracy.
            var meanAccuracy =
                crossValidationResults.Sum(x => x.Metrics.Accuracy) / crossValidationResults.Count();
            Console.WriteLine("----------");
            Console.WriteLine($"Mean accuracy: {meanAccuracy}");
            Console.WriteLine("");

            // Train model based on pipeline.
            // As a train data use comments from 4 first files.
            // As a test data use comments from 5th file (unseen).
            // Then calculate metrics and print confusion matrix and accuracy.
            var trainFiles =
                files.Take(4).ToArray();
            var trainData =
                textLoader.Load(trainFiles);
            var testFiles =
                files.Skip(4).Take(1).ToArray();
            var testData =
                textLoader.Load(testFiles);
            var model =
                pipeline.Fit(trainData);
            var predictions =
                model.Transform(testData);
            var metrics =
                mlContext.BinaryClassification.Evaluate(predictions);
            Console.WriteLine($"Results for unseen data set: {string.Join(',', testFiles)}");
            Console.WriteLine("----------");
            Console.WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());
            Console.WriteLine($"Accuracy: {metrics.Accuracy}");
            Console.WriteLine("");

            // Convert our model to prediction engine and  demonstrate 
            // the outcomes on random samples from test data set.
            var predictionEngine =
                mlContext.Model.CreatePredictionEngine<Comment, Prediction>(model);
            var random = new Random();
            var samples =
                mlContext.Data.CreateEnumerable<Comment>(testData, false)
                .OrderBy(x => random.Next())
                .Take(5);
            foreach(var sample in samples)
            {
                var prediction = predictionEngine.Predict(sample);
                Console.Write("Comment:");
                Console.WriteLine(sample.Content);
                Console.WriteLine("Label: " + (prediction.Label ? "SPAM" : "OK"));
                Console.WriteLine("");
            }
        }
    }
}
