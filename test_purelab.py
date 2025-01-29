def test_imports():
    try:
        import purelab
        print("purelab package found")
        
        import numpy
        print("numpy found")
        
        import pandas
        print("pandas found")
        
        import sklearn
        print("scikit-learn found")
        
        import cleanlab
        print("cleanlab found")
        
        from purelab.utils import data_checker_utils
        print("data_checker_utils found")
        
        print("\nAll required packages are properly installed!")
        return True
    except ImportError as e:
        print(f"Import error: {str(e)}")
        return False

if __name__ == "__main__":
    test_imports() 