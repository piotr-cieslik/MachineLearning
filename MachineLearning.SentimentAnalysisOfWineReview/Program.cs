using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.IO;
using System.Linq;

namespace MachineLearning.SentimentAnalysisOfWineReview
{
    // In this example I'll use regression algorithm to predict number of points (rating) of wine based on:
    // 1) description (only),
    // 2) description, price,
    // 2) description, price and other informations about wine like country etc.
    public static class Program
    {
        static void Main(string[] args)
        {
            var inputDataPath = @"..\..\..\..\IgnoredData\winemag.csv";
            var outputDataPath = @"..\..\..\..\IgnoredData\winemag.predictions.csv";

            var mlContext = new MLContext(0);

            // Because of the file size I don't include data under source controll.
            // The data can be found here https://www.kaggle.com/zynicide/wine-reviews.
            var data =
                mlContext.Data.LoadFromTextFile<Review>(
                    inputDataPath,
                    separatorChar: ',',
                    allowQuoting: true,
                    hasHeader: true);

            // Take first 10 000 date and split it to train and test sets.
            // It's possible to use all data, but training results are almost the same,
            // but it takes a lot longer to train.
            var trainTestData =
                mlContext.Data.TrainTestSplit(
                    mlContext.Data.TakeRows(data, 10000),
                    //data,
                    testFraction: 0.5);

            // Extract features from all variables.
            // The model has almost same result when only description and price is used.
            // The model has decent result when only description is used.
            var pipeline =
                new EstimatorChain<ITransformer>()
                    .Append(
                        mlContext.Transforms.Text.FeaturizeText(outputColumnName: nameof(Review.Country)))
                    .Append(
                        mlContext.Transforms.Text.FeaturizeText(outputColumnName: nameof(Review.Description)))
                    .Append(
                        mlContext.Transforms.Text.FeaturizeText(outputColumnName: nameof(Review.Designation)))
                    .Append(
                        mlContext.Transforms.Text.FeaturizeText(outputColumnName: nameof(Review.Province)))
                    .Append(
                        mlContext.Transforms.Text.FeaturizeText(outputColumnName: nameof(Review.Region1)))
                    .Append(
                        mlContext.Transforms.Text.FeaturizeText(outputColumnName: nameof(Review.Region2)))
                    .Append(
                        mlContext.Transforms.Text.FeaturizeText(outputColumnName: nameof(Review.Variety)))
                    .Append(
                        mlContext.Transforms.Text.FeaturizeText(outputColumnName: nameof(Review.Winery)))
                    .Append(
                        mlContext.Transforms.Concatenate(
                            "Features",
                            nameof(Review.Country),
                            nameof(Review.Description),
                            nameof(Review.Designation),
                            nameof(Review.Province),
                            nameof(Review.Region1),
                            nameof(Review.Region2),
                            nameof(Review.Variety),
                            nameof(Review.Winery),
                            nameof(Review.Price)
                        ))
                    .Append(
                        mlContext.Transforms.CopyColumns("Label", "Points"))
                    .Append(
                        mlContext.Regression.Trainers.FastTree());

            // Train the model.
            var model =
                pipeline
                    .Fit(trainTestData.TrainSet);

            // Use model to predict results of test set.
            var predictions =
                model.Transform(trainTestData.TestSet);

            // Calculate and print metrix.
            var metrics =
                mlContext.Regression.Evaluate(predictions);
            Console.WriteLine($"MeanAbsoluteError:\t{metrics.MeanAbsoluteError}");
            Console.WriteLine($"RootMeanSquaredError:\t{metrics.RootMeanSquaredError}");

            // Create prediction engine and present prediction on few randomly selected samples.
            var predictionEngine =
                mlContext.Model.CreatePredictionEngine<Review, Prediction>(model);
            var random = new Random(0);
            var reviews =
                mlContext.Data.CreateEnumerable<Review>(trainTestData.TestSet, false)
                    .OrderBy(x => random.Next())
                    .Take(5);
            foreach (var review in reviews)
            {
                var prediction = predictionEngine.Predict(review);
                var error = Math.Abs(prediction.Actual - prediction.Predicted);
                Console.WriteLine($"Description: {review.Description}");
                Console.WriteLine($"Predicted: {prediction.Predicted:N2}, Actual: {prediction.Actual:N2}, Error: {error:N2}");
                Console.WriteLine("");
            }

            // Write results to the file CSV files with heders.
            // The columns are: Id, Label, Score.
            using var file = File.Create(outputDataPath);
            mlContext.Data.SaveAsText(
                mlContext.Transforms
                    .SelectColumns(
                        nameof(Review.Id),
                        "Label",
                        "Score")
                    .Fit(predictions)
                    .Transform(predictions),
                file,
                separatorChar: ',',
                headerRow: true,
                schema: false);
        }
    }

    public sealed class Review
    {
        [LoadColumn(0)]
        public string Id { get; set; }

        [LoadColumn(1)]
        public string Country { get; set; }

        [LoadColumn(2)]
        public string Description { get; set; }

        [LoadColumn(3)]
        public string Designation { get; set; }

        [LoadColumn(4)]
        public float Points { get; set; }

        [LoadColumn(5)]
        public float Price { get; set; }

        [LoadColumn(6)]
        public string Province { get; set; }

        [LoadColumn(7)]
        public string Region1 { get; set; }

        [LoadColumn(8)]
        public string Region2 { get; set; }

        [LoadColumn(9)]
        public string Variety { get; set; }

        [LoadColumn(10)]
        public string Winery { get; set; }
    }

    public sealed class Prediction
    {
        [ColumnName("Label")]
        public float Actual { get; set; }

        [ColumnName("Score")]
        public float Predicted { get; set; }
    }
}
