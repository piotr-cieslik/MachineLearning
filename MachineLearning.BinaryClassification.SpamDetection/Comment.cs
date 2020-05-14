using System;
using Microsoft.ML.Data;

namespace MachineLearning.BinaryClassification.SpamDetection
{
    public sealed class Comment
    {
        [LoadColumn(0)]
        public string Id { get; set; }

        [LoadColumn(1)]
        public string Author { get; set; }

        [LoadColumn(2)]
        public DateTime Date { get; set; }

        [LoadColumn(3)]
        public string Content { get; set; }

        [LoadColumn(4)]
        public bool Class { get; set; }
    }
}
