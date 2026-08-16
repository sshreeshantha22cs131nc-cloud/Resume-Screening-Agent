import unittest
from agent import clean_text, combine_scores, compute_tfidf_scores

class TestResumeAgent(unittest.TestCase):

    def test_clean_text_removes_extra_whitespace(self):
        # Setup test data with tabs, newlines, and multiple spaces
        dirty_text = "This   is  a\n\n dirty \t text."
        expected = "This is a dirty text."
        
        # Execute
        result = clean_text(dirty_text)
        
        # Assert
        self.assertEqual(result, expected)

    def test_combine_scores_fallback(self):
        # When LLM score is None, it should return the exact TF-IDF score
        tfidf = 45.12
        llm = None
        
        result = combine_scores(tfidf, llm)
        self.assertEqual(result, 45.12)

    def test_combine_scores_hybrid(self):
        # 40% TF-IDF, 60% LLM
        tfidf = 50.0
        llm = 80.0
        # Expected: 50*0.4 + 80*0.6 = 20 + 48 = 68.0
        
        result = combine_scores(tfidf, llm)
        self.assertEqual(result, 68.0)
        
    def test_compute_tfidf_scores_logic(self):
        # Create a simple JD and resumes that will clearly rank differently
        jd = "python data science machine learning statistics"
        strong_resume = "I am a data scientist experienced in python, machine learning and statistics"
        weak_resume = "I am a mechanic who repairs cars and trucks"
        
        scores = compute_tfidf_scores(jd, [strong_resume, weak_resume])
        
        # We expect 2 scores since we provided 2 resumes
        self.assertEqual(len(scores), 2)
        
        # The strong resume MUST have a noticeably higher tf-idf overlap
        self.assertTrue(scores[0] > scores[1], "The strong resume should score higher than the unrelated one")
        self.assertTrue(scores[1] == 0.0, "The completely unrelated resume should score essentially 0")

if __name__ == '__main__':
    unittest.main()
