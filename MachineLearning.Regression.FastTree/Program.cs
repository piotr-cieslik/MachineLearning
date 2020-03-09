using System;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Trainers.FastTree;

namespace MachineLearning.Regression.FastTree
{
    /// <summary>
    /// Documentations:
    /// - https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.treeextensions.fasttree
    /// - https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.trainers.fasttree.fasttreeregressiontrainer?view=ml-dotnet
    /// </summary>
    public static class Program
    {
        static void Main()
        {
            var mlContext =
                new MLContext(1);

            // Load data.
            var allData =
                mlContext.Data.LoadFromTextFile<WineData>(
                    "wine.data",
                    separatorChar: ',');
            
            // Split data into train and test sets.
            var data =
                mlContext.Data.TrainTestSplit(
                    allData,
                    testFraction: 0.2,
                    seed: 1);
            
            // Define pipeline
            var pipeline =
                mlContext.Transforms.Concatenate(
                    "Features",
                    nameof(WineData.Alcohol),
                    nameof(WineData.MalicAcid),
                    nameof(WineData.Ash),
                    nameof(WineData.AlcalinityOfAash),
                    nameof(WineData.Magnesium),
                    nameof(WineData.TotalPhenols),
                    nameof(WineData.Flavanoids),
                    nameof(WineData.NonflavanoidPhenols),
                    nameof(WineData.Proanthocyanins),
                    nameof(WineData.ColorIntensity),
                    nameof(WineData.Hue),
                    nameof(WineData.OD280_OD315_OfDilutedWines),
                    nameof(WineData.Proline))
                .Append(
                    mlContext.Regression.Trainers.FastTree(
                        numberOfLeaves: 10));
            
            // Create (learn) model on train set.
            var model =
                pipeline.Fit(data.TrainSet);
            
            // Predict labels on test set (unseen).
            var predictions =
                model.Transform(data.TestSet);

            // Create array of true / false, indicating if our prediction matches known label.
            var correctPredictions =
                mlContext.Data.CreateEnumerable<Prediction>(predictions, reuseRowObject: false)
                    .Select(x => new
                    {
                        Label = (int)x.Label, // Labels are integers.
                        Prediction = (int)Math.Round(x.Score) // Round prediction to nearest integer value.
                    })
                    .Select(x => x.Label == x.Prediction)
                    .ToArray();

            var accuracy =
                correctPredictions.Where(x => x).Count() / (float)correctPredictions.Count();
            Console.WriteLine($"Accuracy: {accuracy}");
        }
    }
}