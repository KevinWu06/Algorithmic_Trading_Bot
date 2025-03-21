from portfolio_optimization import MarkowitzOptimizer

INDUSTRY_TICKERS = {
    # Information Technology Sector
    'software': [
        'MSFT',  # Microsoft: Cloud & enterprise leader
        'ORCL',  # Oracle: Database leader
        'CRM',   # Salesforce: CRM leader
        'ADBE',  # Adobe: Creative software leader
        'INTU',  # Intuit: Financial software leader
        'NOW',   # ServiceNow: IT service management
        'SAP',   # SAP: Enterprise software
        'SNOW',  # Snowflake: Data cloud
        'WDAY',  # Workday: HR software
        'PLTR',  # Palantir: Data analytics
    ],
    'semiconductors': [
        'NVDA',  # NVIDIA: AI & GPU leader
        'TSM',   # TSMC: Manufacturing leader
        'AVGO',  # Broadcom: Diverse chips
        'ASML',  # ASML: Equipment monopoly
        'AMD',   # AMD: CPU & GPU innovation
        'QCOM',  # Qualcomm: Mobile chips
        'AMAT',  # Applied Materials: Equipment
        'KLAC',  # KLA: Inspection equipment
        'LRCX',  # Lam Research: Equipment
        'ADI',   # Analog Devices: Analog chips
    ],
    'it_hardware': [
        'AAPL',  # Apple: Consumer electronics
        'HPQ',   # HP: PC & printer
        'DELL',  # Dell: Enterprise hardware
        'CSCO',  # Cisco: Networking
        'ANET',  # Arista: Cloud networking
        'HPE',   # HP Enterprise: Enterprise
        'SMCI',  # Super Micro: Servers
        'NTAP',  # NetApp: Storage
        'JNPR',  # Juniper: Networking
        'FLEX',  # Flex: Manufacturing
    ],

    # Healthcare Sector
    'biotech': [
        'AMGN',  # Amgen: Biotech pioneer
        'REGN',  # Regeneron: Immunology
        'VRTX',  # Vertex: Rare diseases
        'GILD',  # Gilead: Antivirals
        'MRNA',  # Moderna: mRNA tech
        'BIIB',  # Biogen: Neuroscience
        'ILMN',  # Illumina: Gene sequencing
        'SGEN',  # Seagen: Cancer therapy
        'INCY',  # Incyte: Cancer research
        'BMRN',  # BioMarin: Rare diseases
    ],
    'pharma': [
        'JNJ',   # Johnson & Johnson: Healthcare
        'LLY',   # Eli Lilly: Diabetes/cancer
        'PFE',   # Pfizer: Vaccines/therapy
        'MRK',   # Merck: Immunotherapy
        'ABBV',  # AbbVie: Immunology
        'NVS',   # Novartis: Global pharma
        'BMY',   # Bristol Myers: Oncology
        'AZN',   # AstraZeneca: Research
        'GSK',   # GlaxoSmithKline: Vaccines
        'SNY',   # Sanofi: Global healthcare
    ],
    'healthcare_providers': [
        'UNH',   # UnitedHealth: Health insurance
        'CVS',   # CVS Health: Pharmacy/health
        'CI',    # Cigna: Health services
        'HUM',   # Humana: Medicare advantage
        'ANTM',  # Anthem: Health benefits
        'CNC',   # Centene: Government programs
        'MOH',   # Molina: Medicaid/Medicare
        'HCA',   # HCA Healthcare: Hospitals
        'UHS',   # Universal Health: Facilities
        'THC',   # Tenet Healthcare: Hospitals
    ],

    # Financial Sector
    'banks': [
        'JPM',   # JPMorgan: Banking leader
        'BAC',   # Bank of America: Consumer
        'WFC',   # Wells Fargo: Retail
        'MS',    # Morgan Stanley: Investment
        'GS',    # Goldman Sachs: Investment
        'C',     # Citigroup: Global
        'RY',    # Royal Bank of Canada
        'TD',    # TD Bank: North American
        'HSBC',  # HSBC: Global banking
        'BCS',   # Barclays: UK banking
    ],
    'insurance': [
        'BRK.B', # Berkshire: Conglomerate
        'UNH',   # UnitedHealth: Health
        'PGR',   # Progressive: Auto
        'MET',   # MetLife: Life
        'PRU',   # Prudential: Life/retirement
        'AIG',   # AIG: Property/casualty
        'ALL',   # Allstate: Personal
        'MMC',   # Marsh McLennan: Consulting
        'AON',   # Aon: Risk management
        'TRV',   # Travelers: Property/casualty
    ],
    'investment_services': [
        'BLK',   # BlackRock: Asset management
        'SCHW',  # Charles Schwab: Brokerage
        'MS',    # Morgan Stanley: Wealth
        'GS',    # Goldman: Investment
        'BX',    # Blackstone: Private equity
        'KKR',   # KKR: Private equity
        'APO',   # Apollo: Alternative assets
        'CG',    # Carlyle: Private equity
        'TROW',  # T. Rowe Price: Mutual funds
        'BEN',   # Franklin Resources: Asset management
    ],

    # Consumer Discretionary
    'retail': [
        'AMZN',  # Amazon: E-commerce
        'WMT',   # Walmart: Retail
        'HD',    # Home Depot: Home improvement
        'COST',  # Costco: Wholesale
        'TGT',   # Target: Retail
        'LOW',   # Lowe's: Home improvement
        'TJX',   # TJX: Off-price retail
        'ROST',  # Ross Stores: Off-price
        'BBY',   # Best Buy: Electronics
        'DG',    # Dollar General: Discount
    ],

    'automotive': [
        'TSLA',  # Tesla: EV leader
        'TM',    # Toyota: Global leader
        'F',     # Ford: Legacy/EV transition
        'GM',    # General Motors: Legacy/EV
        'STLA',  # Stellantis: Global auto
        'BMW.DE',# BMW: Luxury leader
        'HMC',   # Honda: Global mobility
        'RACE',  # Ferrari: Luxury performance
        'RIVN',  # Rivian: EV startup
        'LCID',  # Lucid: Luxury EV
    ],
    'luxury_goods': [
        'LVMH.PA', # LVMH: Luxury conglomerate
        'NKE',   # Nike: Athletic wear
        'ADDYY', # Adidas: Sportswear
        'TPR',   # Tapestry: Accessories
        'RL',    # Ralph Lauren: Fashion
        'CPRI',  # Capri Holdings: Fashion
        'VFC',   # VF Corp: Apparel
        'LULU',  # Lululemon: Athleisure
        'DECK',  # Deckers: Footwear
        'PVH',   # PVH Corp: Apparel
    ],

    # Communication Services
    'telecom': [
        'T',     # AT&T: Telecom giant
        'VZ',    # Verizon: Wireless leader
        'TMUS',  # T-Mobile: Mobile carrier
        'CMCSA', # Comcast: Cable/internet
        'CHTR',  # Charter: Cable services
        'AMT',   # American Tower: Infrastructure
        'CCI',   # Crown Castle: 5G infrastructure
        'VOD',   # Vodafone: Global telecom
        'TEF',   # Telefonica: Global telecom
        'BCE',   # BCE: Canadian telecom
    ],
    'media_entertainment': [
        'GOOGL', # Alphabet: Digital ads/media
        'META',  # Meta: Social media
        'DIS',   # Disney: Entertainment
        'NFLX',  # Netflix: Streaming
        'WBD',   # Warner Bros Discovery
        'PARA',  # Paramount Global
        'SPOT',  # Spotify: Audio streaming
        'EA',    # Electronic Arts: Gaming
        'ATVI',  # Activision: Gaming
        'TTWO',  # Take-Two: Gaming
    ],

    # Energy Sector
    'oil_gas': [
        'XOM',   # ExxonMobil: Oil major
        'CVX',   # Chevron: Integrated
        'COP',   # ConocoPhillips: E&P
        'SLB',   # Schlumberger: Services
        'EOG',   # EOG Resources: Shale
        'PXD',   # Pioneer: Permian
        'MPC',   # Marathon: Refining
        'PSX',   # Phillips 66: Refining
        'VLO',   # Valero: Refining
        'OXY',   # Occidental: E&P
    ],
    'renewable_energy': [
        'NEE',   # NextEra: Utility/renewables
        'ENPH',  # Enphase: Solar tech
        'SEDG',  # SolarEdge: Solar
        'FSLR',  # First Solar: Solar panels
        'RUN',   # Sunrun: Residential solar
        'BE',    # Bloom Energy: Fuel cells
        'PLUG',  # Plug Power: Hydrogen
        'NOVA',  # Sunnova: Solar
        'STEM',  # Stem: Energy storage
        'CHPT',  # ChargePoint: EV charging
    ],

    # Materials Sector
    'chemicals': [
        'LIN',   # Linde: Industrial gases
        'APD',   # Air Products: Gases
        'SHW',   # Sherwin-Williams: Coatings
        'DD',    # DuPont: Specialty chemicals
        'ECL',   # Ecolab: Water/hygiene
        'PPG',   # PPG Industries: Coatings
        'ALB',   # Albemarle: Lithium
        'IFF',   # International Flavors
        'CE',    # Celanese: Specialty materials
        'EMN',   # Eastman Chemical
    ],
    'metals_mining': [
        'BHP',   # BHP: Diversified mining
        'RIO',   # Rio Tinto: Mining
        'VALE',  # Vale: Iron ore
        'FCX',   # Freeport-McMoRan: Copper
        'NEM',   # Newmont: Gold
        'AA',    # Alcoa: Aluminum
        'NUE',   # Nucor: Steel
        'STLD',  # Steel Dynamics
        'SCCO',  # Southern Copper
        'X',     # US Steel
    ],

    # Industrials Sector
    'aerospace_defense': [
        'LMT',   # Lockheed Martin: Defense
        'RTX',   # Raytheon: Aerospace
        'BA',    # Boeing: Aircraft
        'NOC',   # Northrop Grumman: Defense
        'GD',    # General Dynamics: Defense
        'LHX',   # L3Harris: Defense tech
        'HEI',   # HEICO: Aerospace
        'TDG',   # TransDigm: Aircraft parts
        'HII',   # Huntington Ingalls: Ships
        'LDOS',  # Leidos: Defense IT
    ],
    'machinery': [
        'CAT',   # Caterpillar: Heavy equipment
        'DE',    # Deere: Agriculture
        'ITW',   # Illinois Tool Works
        'ETN',   # Eaton: Power management
        'EMR',   # Emerson: Automation
        'ROK',   # Rockwell: Automation
        'PH',    # Parker-Hannifin: Motion
        'DOV',   # Dover: Industrial
        'IR',    # Ingersoll Rand
        'CMI',   # Cummins: Engines
    ],

    # Real Estate
    'reits': [
        'PLD',   # Prologis: Logistics
        'AMT',   # American Tower: Telecom
        'EQIX',  # Equinix: Data centers
        'CCI',   # Crown Castle: Wireless
        'SPG',   # Simon Property: Malls
        'WELL',  # Welltower: Healthcare
        'AVB',   # AvalonBay: Apartments
        'EQR',   # Equity Residential
        'DLR',   # Digital Realty: Data
        'PSA',   # Public Storage
    ],

    # Consumer Staples
    'food_beverage': [
        'KO',    # Coca-Cola: Beverages
        'PEP',   # PepsiCo: Snacks/beverages
        'MDLZ',  # Mondelez: Snacks
        'PM',    # Philip Morris: Tobacco
        'KHC',   # Kraft Heinz: Food
        'STZ',   # Constellation: Alcohol
        'MO',    # Altria: Tobacco
        'EL',    # Estée Lauder: Beauty
        'KMB',   # Kimberly-Clark: Personal care
        'CL',    # Colgate: Personal care
    ],
    'household_products': [
        'PG',    # Procter & Gamble: Consumer
        'COST',  # Costco: Retail
        'WMT',   # Walmart: Retail
        'TGT',   # Target: Retail
        'KR',    # Kroger: Grocery
        'SYY',   # Sysco: Food distribution
        'GIS',   # General Mills: Food
        'HSY',   # Hershey: Confectionery
        'K',     # Kellogg: Food
        'CAG',   # ConAgra: Food
    ],

    # Utilities
    'utilities': [
        'NEE',   # NextEra: Renewable/utility
        'DUK',   # Duke: Electric
        'SO',    # Southern Company
        'D',     # Dominion Energy
        'AEP',   # American Electric Power
        'SRE',   # Sempra: Energy infra
        'EXC',   # Exelon: Utility
        'PCG',   # PG&E: Utility
        'XEL',   # Xcel Energy
        'WEC',   # WEC Energy
    ],

    # Technology Services
    'tech_services': [
        'UBER',  # Uber: Ride sharing
        'DASH',  # DoorDash: Food delivery
        'ABNB',  # Airbnb: Travel platform
        'SQ',    # Block: Fintech
        'PYPL',  # PayPal: Payments
        'ADYEY', # Adyen: Payments
        'SHOP',  # Shopify: E-commerce
        'ZM',    # Zoom: Video communications
        'TWLO',  # Twilio: Communications
        'NET',   # Cloudflare: Security
    ]
}

def get_top_companies_by_industry(industry, limit=50):
    """
    Get top companies based on multiple factors for a specific GICS industry
    """
    if industry.lower() not in INDUSTRY_TICKERS:
        raise ValueError(f"Industry '{industry}' not supported. Choose from: {list(INDUSTRY_TICKERS.keys())}")
    
    return INDUSTRY_TICKERS[industry.lower()][:limit]

if __name__ == "__main__":
    # Get available industries
    print("\nAvailable industries:")
    industries = list(enumerate(sorted(INDUSTRY_TICKERS.keys()), 1))
    
    for i, industry in industries:
        print(f"{i}. {industry.replace('_', ' ').title()}")
    
    # Get user choice
    choice = input(f"\nChoose an industry (1-{len(industries)}): ")
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(industries):
            industry = industries[idx][1]
        else:
            print("Invalid choice. Defaulting to software sector.")
            industry = 'software'
    except ValueError:
        print("Invalid input. Defaulting to software sector.")
        industry = 'software'
    
    # Get tickers for chosen industry
    tickers = get_top_companies_by_industry(industry, limit=50)
    print(f"\nAnalyzing top companies in {industry.replace('_', ' ').title()} sector...")
    print("Tickers:", tickers)
    
    # Create optimizer instance
    optimizer = MarkowitzOptimizer(tickers)
    
    try:
        # Fetch data for the last year
        returns = optimizer.fetch_data()
        
        # Find optimal portfolio
        optimal_weights, exp_return, volatility, sharpe = optimizer.optimize_portfolio()
        
        # Print results for companies with non-zero weights (>0.1%)
        print(f"\nOptimal Portfolio Weights for {industry.replace('_', ' ').title()} Sector (>0.1%):")
        for ticker, weight in zip(tickers, optimal_weights):
            if weight > 0.001:  # Only show weights > 0.1%
                print(f"{ticker}: {weight:.4f}")
        
        print(f"\nExpected Annual Return: {exp_return:.4f}")
        print(f"Annual Volatility: {volatility:.4f}")
        print(f"Sharpe Ratio: {sharpe:.4f}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Try reducing the number of companies or checking data availability.") 