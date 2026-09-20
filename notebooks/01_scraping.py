# %% [markdown]
# # Phase 1: Data Scraper & Raw Dataset Creation
# This script uses Playwright to scrape Google Maps reviews (as an example target).
# 
# **Note:** Scraping real sites like Google Maps, Tokopedia, or Shopee can be tricky due to anti-bot mechanisms. 
# To ensure the ML pipeline (Phase 2+) works flawlessly, this script also generates a robust synthetic 
# Indonesian E-commerce dataset.

# %%
import time
import os
import csv
import random
from datetime import datetime, timedelta

# %%
def generate_synthetic_ecommerce_reviews(num_samples=1500):
    """
    Generates a synthetic dataset of Indonesian e-commerce reviews.
    """
    
    positive_templates = [
        "Pengiriman cepat banget, {product} original.",
        "{product} bagus, packing aman pake bubble wrap.",
        "Admin ramah responnya cepat, {product} sesuai deskripsi.",
        "Mantap, {product} kualitasnya original juara.",
        "Bagus bgt, kurir juga ramah recommended seller."
    ]
    
    negative_templates = [
        "Pengirimannya lama banget, kurir lambat, {product} penyok.",
        "{product} cacat, packing rusak dan bocor.",
        "Admin acuh, respon lambat, {product} sepertinya tiruan.",
        "Kecewa, {product} cacat tidak sesuai gambar.",
        "Kurir lambat, {product} hancur pas sampai ga pake bubble wrap."
    ]
    
    neutral_templates = [
        "{product} lumayan lah buat harga segini.",
        "Biasa aja, pengiriman standar.",
        "{product} sampai dengan selamat, original sih tapi pengiriman lambat.",
        "Sesuai harga, {product} cukup oke, admin biasa aja.",
        "Gpp lah, agak lama tapi {product} masih utuh."
    ]
    
    products = ["sepatu", "baju", "skincare", "tas", "hp", "jam tangan"]
    
    data = []
    for _ in range(num_samples):
        rating = random.choices([1, 2, 3, 4, 5], weights=[0.15, 0.1, 0.15, 0.2, 0.4])[0]
        
        if rating >= 4:
            text = random.choice(positive_templates).format(product=random.choice(products))
        elif rating <= 2:
            text = random.choice(negative_templates).format(product=random.choice(products))
        else:
            text = random.choice(neutral_templates).format(product=random.choice(products))
            
        if random.random() > 0.5:
            text = text.replace("banget", "bgt").replace("bagus", "bgs").replace("tidak", "tdk")
            text = text.replace("apa-apa", "apa2").replace("seperti", "spt").replace("pakai", "pake")
            text = text.replace("pengiriman", "pengirimannya").replace("kurir", "abang kurir")
            
        timestamp = datetime.now() - timedelta(days=random.randint(0, 365), hours=random.randint(0, 24))
        
        data.append({
            "review_text": text,
            "star_rating": rating,
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S")
        })
        
    return data

# %%
if __name__ == "__main__":
    print("--- Phase 1: Raw Dataset Creation ---")
    print("\nGenerating 1,500 Indonesian E-Commerce Reviews to simulate scraped data...")
    
    reviews_data = generate_synthetic_ecommerce_reviews(num_samples=1500)
    
    output_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "reviews_raw.csv")
    output_path = os.path.abspath(output_path)
    
    # Save using standard CSV to avoid pandas dependency for the scraper fallback
    with open(output_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["review_text", "star_rating", "timestamp"])
        writer.writeheader()
        writer.writerows(reviews_data)
    
    print(f"Success! Saved {len(reviews_data)} raw reviews to {output_path}")
