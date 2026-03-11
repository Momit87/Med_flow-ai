"""
Setup verification script for MedFlow AI.
Checks that all components are properly configured.
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def check_imports():
    """Verify all required packages are installed."""
    print("\n" + "=" * 70)
    print("CHECKING PYTHON PACKAGES")
    print("=" * 70)

    required_packages = [
        ("langchain", "LangChain"),
        ("langgraph", "LangGraph"),
        ("langchain_google_genai", "Google Gemini Integration"),
        ("chromadb", "ChromaDB"),
        ("streamlit", "Streamlit"),
        ("pydantic", "Pydantic"),
        ("dotenv", "Python-dotenv"),
    ]

    all_good = True
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"✓ {name:30s} installed")
        except ImportError:
            print(f"✗ {name:30s} MISSING")
            all_good = False

    return all_good


def check_env_vars():
    """Check environment variables."""
    print("\n" + "=" * 70)
    print("CHECKING ENVIRONMENT VARIABLES")
    print("=" * 70)

    from dotenv import load_dotenv
    load_dotenv()

    required_vars = [
        ("GOOGLE_API_KEY", True),
        ("GROQ_API_KEY", False),
        ("LANGCHAIN_TRACING_V2", False),
        ("LANGCHAIN_API_KEY", False),
    ]

    all_good = True
    for var, required in required_vars:
        value = os.getenv(var)
        if value:
            masked = value[:8] + "..." if len(value) > 8 else "***"
            print(f"✓ {var:25s} = {masked}")
        else:
            status = "REQUIRED" if required else "optional"
            print(f"{'✗' if required else '○'} {var:25s} not set ({status})")
            if required:
                all_good = False

    return all_good


def check_src_modules():
    """Check that all source modules load correctly."""
    print("\n" + "=" * 70)
    print("CHECKING SOURCE MODULES")
    print("=" * 70)

    modules = [
        ("src.config", "Configuration"),
        ("src.state", "State Schemas"),
        ("src.rag", "RAG Manager"),
        ("src.graph", "Main Graph"),
    ]

    all_good = True
    for module, name in modules:
        try:
            __import__(module)
            print(f"✓ {name:30s} loads successfully")
        except Exception as e:
            print(f"✗ {name:30s} ERROR: {str(e)[:50]}")
            all_good = False

    return all_good


def check_chromadb():
    """Check ChromaDB setup and collections."""
    print("\n" + "=" * 70)
    print("CHECKING CHROMADB")
    print("=" * 70)

    try:
        from src.rag import rag_manager

        collections = {
            "drug_interactions": "Drug Interaction Database",
            "clinical_guidelines": "Clinical Guidelines",
            "patient_education": "Patient Education"
        }

        all_good = True
        for coll_name, display_name in collections.items():
            try:
                count = rag_manager.get_collection_count(coll_name)
                status = "✓" if count > 0 else "○"
                print(f"{status} {display_name:30s} {count:3d} documents")
                if count == 0:
                    print(f"  ⚠ Collection is empty. Run: python scripts/ingest_medical_data.py")
            except Exception as e:
                print(f"✗ {display_name:30s} ERROR: {str(e)[:40]}")
                all_good = False

        return all_good

    except Exception as e:
        print(f"✗ Failed to initialize RAG manager: {e}")
        return False


def check_graph():
    """Test that the graph can be compiled."""
    print("\n" + "=" * 70)
    print("CHECKING LANGGRAPH COMPILATION")
    print("=" * 70)

    try:
        from src.graph import app
        print("✓ Graph compiles successfully")
        print(f"✓ Graph has {len(app.get_graph().nodes)} nodes")
        return True
    except Exception as e:
        print(f"✗ Graph compilation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all verification checks."""
    print("\n" + "=" * 70)
    print("MedFlow AI - Setup Verification")
    print("=" * 70)

    results = {
        "Packages": check_imports(),
        "Environment": check_env_vars(),
        "Modules": check_src_modules(),
        "ChromaDB": check_chromadb(),
        "Graph": check_graph(),
    }

    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)

    all_passed = True
    for check, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8s} {check}")
        if not passed:
            all_passed = False

    print("=" * 70)

    if all_passed:
        print("\n🎉 All checks passed! System is ready.")
        print("\nNext steps:")
        print("1. If ChromaDB is empty, run: python scripts/ingest_medical_data.py")
        print("2. Launch the app: streamlit run app.py")
    else:
        print("\n⚠️  Some checks failed. Please review the errors above.")
        print("\nCommon fixes:")
        print("- Install missing packages: pip install -r requirements.txt")
        print("- Set up .env file: cp .env.example .env")
        print("- Add your GOOGLE_API_KEY to .env")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
