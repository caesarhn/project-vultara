import json
import csv
import pandas as pd
import re
from typing import Dict, List, Any, Union
import logging
from collections.abc import MutableMapping
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CVEJsonFlattener:
    def __init__(self):
        self.processed_data = []
    
    def clean_text(self, text: str) -> str:
        """Bersihkan teks dari simbol dan normalisasi"""
        if not text or text == "":
            return ""
        
        # Hilangkan karakter khusus tapi pertahankan huruf, angka, spasi, dan beberapa karakter umum
        text = re.sub(r'[^\w\s\.\-@/:]', ' ', str(text))
        
        # Ganti multiple spaces dengan single space
        text = re.sub(r'\s+', ' ', text)
        
        # Convert ke lowercase dan strip
        text = text.lower().strip()
        
        return text
    
    def flatten_dict(self, d: Dict, parent_key: str = '', sep: str = ' ') -> Dict[str, Any]:
        """
        Flatten nested dictionary recursively
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            
            if isinstance(v, MutableMapping):
                # Recursively flatten nested dictionaries
                items.extend(self.flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                # Process lists - join with space if strings, otherwise process each item
                if v and all(isinstance(item, str) for item in v):
                    # Join list of strings
                    items.append((new_key, ' '.join(v)))
                elif v:
                    # Process list of objects/dictionaries
                    for i, item in enumerate(v):
                        if isinstance(item, MutableMapping):
                            items.extend(self.flatten_dict(item, f"{new_key}_{i}", sep=sep).items())
                        else:
                            items.append((f"{new_key}_{i}", str(item)))
            else:
                # Simple values
                items.append((new_key, v))
        
        return dict(items)
    
    def extract_cve_id(self, cve_data: Dict) -> str:
        """Extract CVE ID from CVE data"""
        try:
            if 'CVE_data_meta' in cve_data and 'ID' in cve_data['CVE_data_meta']:
                return cve_data['CVE_data_meta']['ID']
            return "unknown"
        except:
            return "unknown"
    
    def process_cve_item(self, cve_item: Dict) -> Dict[str, str]:
        """Process individual CVE item and create flattened document"""
        try:
            cve_data = cve_item.get('cve', {})
            cve_id = self.extract_cve_id(cve_data)
            
            if cve_id == "unknown":
                logger.warning("Skipping CVE item with unknown ID")
                return None
            
            # Flatten the entire CVE item
            flattened_data = self.flatten_dict(cve_item)
            
            # Clean all values and remove empty ones
            cleaned_data = {}
            for key, value in flattened_data.items():
                cleaned_value = self.clean_text(str(value))
                if cleaned_value:  # Only include non-empty values
                    cleaned_data[key] = cleaned_value
            
            # Create document text by combining all key-value pairs
            document_parts = []
            for key, value in cleaned_data.items():
                # Skip very long values that might be noisy
                if len(value) > 500:
                    continue
                document_parts.append(f"{key} {value}")
            
            document_text = " ".join(document_parts)
            
            # Additional cleaning for the final document
            document_text = self.clean_text(document_text)
            
            return {
                'id_cve': cve_id.lower(),
                'dokument': document_text
            }
            
        except Exception as e:
            logger.error(f"Error processing CVE item: {e}")
            return None

    def process_json_file(self, json_file_path: str) -> List[Dict]:
        """Process entire JSON file"""
        try:
            logger.info(f"Loading JSON file: {json_file_path}")
            
            with open(json_file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            cve_items = data.get('CVE_Items', [])
            logger.info(f"Found {len(cve_items)} CVE items to process")
            
            processed_count = 0
            for i, item in enumerate(cve_items):
                if i % 1000 == 0:  # Progress logging
                    logger.info(f"Processed {i}/{len(cve_items)} items")
                
                processed_item = self.process_cve_item(item)
                if processed_item:
                    self.processed_data.append(processed_item)
                    processed_count += 1
            
            logger.info(f"Successfully processed {processed_count} CVE items")
            return self.processed_data
            
        except Exception as e:
            logger.error(f"Error processing JSON file: {e}")
            return []
    
    def save_to_csv(self, output_file: str):
        """Save processed data to CSV"""
        if not self.processed_data:
            logger.warning("No data to save")
            return
        
        try:
            df = pd.DataFrame(self.processed_data)
            df.to_csv(output_file, index=False, encoding='utf-8')
            logger.info(f"Data saved to {output_file}")
            
            # Print sample of the data
            print(f"\n=== SAMPLE DATA (First 3 rows) ===")
            for i, row in df.head(3).iterrows():
                print(f"\n--- CVE {i+1} ---")
                print(f"ID: {row['id_cve']}")
                print(f"Document preview: {row['dokument'][:200]}...")
                
        except Exception as e:
            logger.error(f"Error saving to CSV: {e}")

# Advanced version with better field processing
class AdvancedCVEJsonFlattener(CVEJsonFlattener):
    def __init__(self):
        super().__init__()
    
    def process_cve_item_advanced(self, cve_item: Dict) -> Dict[str, str]:
        """Advanced processing with better field handling"""
        try:
            cve_data = cve_item.get('cve', {})
            cve_id = self.extract_cve_id(cve_data)
            
            if cve_id == "unknown":
                return None
            
            # Process different sections with custom logic
            document_parts = []
            
            # 1. Basic CVE info
            basic_info = self._extract_basic_info(cve_data, cve_id)
            if basic_info:
                document_parts.append(basic_info)
            
            # 2. Problem type (CWE)
            problem_info = self._extract_problem_type(cve_data.get('problemtype', {}))
            if problem_info:
                document_parts.append(problem_info)
            
            # 3. Description
            description_info = self._extract_description(cve_data.get('description', {}))
            if description_info:
                document_parts.append(description_info)
            
            # 4. References
            references_info = self._extract_references(cve_data.get('references', {}))
            if references_info:
                document_parts.append(references_info)
            
            # 5. Configurations (CPE)
            config_info = self._extract_configurations(cve_item.get('configurations', {}))
            if config_info:
                document_parts.append(config_info)
            
            # 6. Impact (CVSS)
            impact_info = self._extract_impact(cve_item.get('impact', {}))
            if impact_info:
                document_parts.append(impact_info)
            
            # 7. Dates
            dates_info = self._extract_dates(cve_item)
            if dates_info:
                document_parts.append(dates_info)
            
            # 8. Flatten the rest for any missing fields
            flattened_rest = self.flatten_dict(cve_item)
            rest_info = self._process_flattened_data(flattened_rest)
            if rest_info:
                document_parts.append(rest_info)
            
            # Combine all parts
            document_text = " ".join(document_parts)
            document_text = self.clean_text(document_text)
            
            return {
                'id_cve': cve_id.lower(),
                'dokument': document_text
            }
            
        except Exception as e:
            logger.error(f"Error in advanced processing: {e}")
            return None
    
    def _extract_basic_info(self, cve_data: Dict, cve_id: str) -> str:
        """Extract basic CVE information"""
        parts = []
        
        meta = cve_data.get('CVE_data_meta', {})
        assigner = meta.get('ASSIGNER', '')
        if assigner:
            parts.append(f"assigner {assigner}")
        
        data_type = cve_data.get('data_type', '')
        if data_type:
            parts.append(f"data_type {data_type}")
        
        data_format = cve_data.get('data_format', '')
        if data_format:
            parts.append(f"data_format {data_format}")
        
        return " ".join(parts)
    
    def _extract_problem_type(self, problemtype: Dict) -> str:
        """Extract problem type information"""
        parts = []
        
        if not problemtype or 'problemtype_data' not in problemtype:
            return ""
        
        for problem_data in problemtype.get('problemtype_data', []):
            for desc in problem_data.get('description', []):
                if desc.get('lang') == 'en':
                    cwe_value = desc.get('value', '')
                    if cwe_value:
                        parts.append(f"cwe {cwe_value}")
        
        return " ".join(parts)
    
    def _extract_description(self, description: Dict) -> str:
        """Extract description information"""
        parts = []
        
        if not description or 'description_data' not in description:
            return ""
        
        for desc in description.get('description_data', []):
            if desc.get('lang') == 'en':
                desc_value = desc.get('value', '')
                if desc_value:
                    parts.append(f"description {desc_value}")
        
        return " ".join(parts)
    
    def _extract_references(self, references: Dict) -> str:
        """Extract references information"""
        parts = []
        
        if not references or 'reference_data' not in references:
            return ""
        
        for ref in references.get('reference_data', []):
            url = ref.get('url', '')
            if url:
                parts.append(f"reference_url {url}")
            
            name = ref.get('name', '')
            if name:
                parts.append(f"reference_name {name}")
            
            tags = ref.get('tags', [])
            for tag in tags:
                if tag:
                    parts.append(f"reference_tag {tag}")
        
        return " ".join(parts)
    
    def _extract_configurations(self, configurations: Dict) -> str:
        """Extract configuration information"""
        parts = []
        
        if not configurations or 'nodes' not in configurations:
            return ""
        
        for node in configurations.get('nodes', []):
            operator = node.get('operator', '')
            if operator:
                parts.append(f"operator {operator}")
            
            for cpe_match in node.get('cpe_match', []):
                cpe_uri = cpe_match.get('cpe23Uri', '')
                if cpe_uri:
                    parts.append(f"cpe {cpe_uri}")
                
                vulnerable = cpe_match.get('vulnerable', False)
                if vulnerable:
                    parts.append("vulnerable true")
                
                # Version information
                version_start = cpe_match.get('versionStartIncluding', '')
                if version_start:
                    parts.append(f"version_start {version_start}")
                
                version_end = cpe_match.get('versionEndIncluding', '')
                if version_end:
                    parts.append(f"version_end {version_end}")
        
        return " ".join(parts)
    
    def _extract_impact(self, impact: Dict) -> str:
        """Extract impact and CVSS information"""
        parts = []
        
        if not impact:
            return ""
        
        # CVSS v3
        if 'baseMetricV3' in impact:
            cvss_v3 = impact['baseMetricV3'].get('cvssV3', {})
            if cvss_v3:
                version = cvss_v3.get('version', '')
                if version:
                    parts.append(f"cvss_version {version}")
                
                vector = cvss_v3.get('vectorString', '')
                if vector:
                    parts.append(f"cvss_vector {vector}")
                
                score = cvss_v3.get('baseScore', '')
                if score:
                    parts.append(f"cvss_score {score}")
                
                severity = cvss_v3.get('baseSeverity', '')
                if severity:
                    parts.append(f"cvss_severity {severity}")
        
        # CVSS v2
        if 'baseMetricV2' in impact:
            cvss_v2 = impact['baseMetricV2'].get('cvssV2', {})
            if cvss_v2:
                score = cvss_v2.get('baseScore', '')
                if score:
                    parts.append(f"cvss_v2_score {score}")
        
        return " ".join(parts)
    
    def _extract_dates(self, cve_item: Dict) -> str:
        """Extract date information"""
        parts = []
        
        published = cve_item.get('publishedDate', '')
        if published:
            parts.append(f"published {published}")
        
        last_modified = cve_item.get('lastModifiedDate', '')
        if last_modified:
            parts.append(f"modified {last_modified}")
        
        return " ".join(parts)
    
    def _process_flattened_data(self, flattened_data: Dict) -> str:
        """Process remaining flattened data"""
        parts = []
        
        for key, value in flattened_data.items():
            cleaned_value = self.clean_text(str(value))
            if cleaned_value and len(cleaned_value) < 500:  # Skip very long values
                # Skip already processed fields
                if not any(field in key for field in ['cve_data_meta', 'problemtype', 'description', 
                                                     'references', 'configurations', 'impact']):
                    parts.append(f"{key} {cleaned_value}")
        
        return " ".join(parts)

# Demo and usage
def main():
    """Main function to demonstrate the CVE JSON to CSV conversion"""
    
    # Choose which processor to use
    use_advanced = True  # Set to False for basic processor
    
    if use_advanced:
        processor = AdvancedCVEJsonFlattener()
        processor_name = "Advanced"
    else:
        processor = CVEJsonFlattener()
        processor_name = "Basic"
    
    print(f"Using {processor_name} CVE JSON Flattener")
    print("=" * 50)
    
    # Process JSON file
    json_file = "nvdcve-1.1-2024.json"  # Ganti dengan path file JSON Anda
    
    try:
        processed_data = processor.process_json_file(json_file)
        
        if processed_data:
            # Save to CSV
            output_file = f"cve_flattened_{processor_name.lower()}.csv"
            processor.save_to_csv(output_file)
            
            # Print statistics
            print(f"\n=== PROCESSING STATISTICS ===")
            print(f"Total CVEs processed: {len(processed_data)}")
            
            if processed_data:
                avg_doc_length = sum(len(item['dokument']) for item in processed_data) / len(processed_data)
                print(f"Average document length: {avg_doc_length:.0f} characters")
                print(f"Sample CVE IDs: {[item['id_cve'] for item in processed_data[:3]]}")
        
        else:
            print("No data was processed. Please check your JSON file.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":

    data_csv = [
        ["id_cve", "documents"]
    ]
    # Run demo with sample data
    with open('nvdcve-1.1-2024.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    advanced_processor = AdvancedCVEJsonFlattener()
    for item in tqdm(data["CVE_Items"], desc="converting data.."):
        res = advanced_result = advanced_processor.process_cve_item_advanced(item)
        data_csv.append([res["id_cve"], res["dokument"]])


    # Tulis file CSV
    filename = "data_bm25.csv"
    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        for row in tqdm(data_csv, desc="generating csv"):
            writer.writerow(row)
    
    print("\n" + "="*60)
    
    # Run main processing (comment out if you don't have the JSON file yet)
    # main()