using System;
using Microsoft.ML;

namespace MachineLearning.BinaryClassification.LinearSvm
{
    public class Program
    {
        /// <summary>
        /// Documentations:
        /// - https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.standardtrainerscatalog.linearsvm
        /// Data source:
        /// - http://archive.ics.uci.edu/ml/datasets/Iris
        /// </summary>
        public static void Main()
        {
            var mlContext =
                new MLContext(seed: 0);

            // Load and split data into train and test.
            var irisData =
                mlContext.Data.LoadFromTextFile<IrisData>(
                    "iris.data",
                    separatorChar: ',');
            var data =
                mlContext.Data.TrainTestSplit(
                    irisData,
                    testFraction: 0.2);

            // Define the pipeline:
            // - Convert class of a flower from string to one of 3 boolean values.
            // - Define the features.
            // - Define SVM trainer for one of the flower class: setosa.
            var pipeline =
                mlContext.Transforms.CustomMapping<IrisData, IrisDataCalculated>(
                    (x, y) =>
                    {
                        y.Setosa = x.Class.Contains("setosa");
                        y.Versicolor = x.Class.Contains("versicolor");
                        y.Virginica = x.Class.Contains("virginica");
                    },
                    contractName: default)
                .Append(
                    mlContext.Transforms.Concatenate(
                        "Features",
                        nameof(IrisData.SepalLength),
                        nameof(IrisData.SepalWidth),
                        nameof(IrisData.PetalLength),
                        nameof(IrisData.PetalWidth)))
                .Append(
                    mlContext.BinaryClassification.Trainers.LinearSvm(
                        labelColumnName: nameof(IrisDataCalculated.Setosa),
                        featureColumnName: "Features",
                        numberOfIterations: 5)); // Adjusted experimentally

            // Train the model.
            var model =
                pipeline.Fit(data.TrainSet);

            // Run the model on test data set.
            var prediction =
                model.Transform(data.TestSet);

            // Calculate metrics of the model.
            var metrics =
                mlContext.BinaryClassification.EvaluateNonCalibrated(
                    prediction,
                    labelColumnName: nameof(IrisDataCalculated.Setosa));

            // Print metrics.
            Console.WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());
            Console.WriteLine($"Accuracy: {metrics.Accuracy}");
        }
    }
}