/**
 * AI Risk Manager — Fraud Protection Concept Specifications
 * 
 * Detailed architectural specifications for fraud protection systems
 * currently in research / conceptual phase (not yet implemented in production).
 * 
 * All data points, pipeline steps, and required inputs are documented with
 * plain-language precision matching the platform's financial-trust standards.
 */

const FRAUD_CONCEPTS = {
  qr_overlay_image: {
    id: 'qr_overlay_image',
    title: 'QR Overlay Attack (Image ML)',
    short_name: 'QR Overlay',
    category: 'UPI Payment Defense',
    status: 'Concept / Not yet implemented',
    target_metric: 'Recall ≥ 0.90 @ FPR ≤ 0.1%',
    estimated_timeline: '4 weeks engineering + field data collection',
    primary_loss_owner: 'Retail Merchants & Offline Payers',
    
    fraud_pattern: {
      summary: 'Attackers print adhesive stickers bearing their own UPI QR code and paste them directly over genuine merchant counter standees, silently diverting in-person customer payments to the attacker\'s bank account.',
      mechanics: 'Fraudsters visit busy retail counters (grocery stores, fuel stations, restaurants) during peak hours. They paste a thin, precisely dimensioned sticker with their own VPA (Virtual Payment Address) over the merchant\'s static BharatQR or UPI QR standee. Because the payment completes normally on the customer\'s mobile app without an immediate soundbox alert on the merchant\'s device, several transactions pass before the store owner notices missing funds.',
      financial_impact: 'Immediate direct revenue theft for offline merchants. Average loss ranges from ₹2,000 to ₹35,000 per compromised counter before detection. Payer dispute resolution is prolonged because the payer authorized the transfer to an invalid payee without merchant acknowledgement.',
      why_existing_rules_fail: 'Transaction velocity and device integrity rules see completely normal customer phones making valid peer-to-merchant payments. The UPI backend processes the payment as a legitimate transaction to the attacker\'s handle because the payment network has no insight into the physical QR surface scanned.'
    },

    detector_function: {
      summary: 'A computer-vision sub-detector running at the payment application edge and merchant soundbox validation layer to detect physical sticker edges, perspective mismatches, and cryptographic hash deviations before settlement.',
      pipeline_steps: [
        {
          step: 1,
          name: 'Optical Capture & Normalization',
          description: 'The scanning camera captures the QR viewfinder frame. Bilinear perspective transformation rectifies the QR square to standard 512×512 resolution, normalizing for ambient lighting and skew angle.'
        },
        {
          step: 2,
          name: 'Physical Sticker Edge Segmentation (YOLOv8-seg)',
          description: 'A lightweight edge segmentation model inspects the perimeter of the QR square for multi-layer depth cues, adhesive border shadows, paper-grain discontinuities, or peel-and-stick alignment offsets.'
        },
        {
          step: 3,
          name: 'Perceptual Hash (pHash) Verification',
          description: 'Calculates a discrete cosine transform (DCT) based perceptual hash of the scanned QR visual pattern and compares Hamming distance against the merchant\'s master QR template stored in the registry.'
        },
        {
          step: 4,
          name: 'Geofence & VPA Cross-Reference',
          description: 'Decodes the embedded UPI URI (`upi://pay?pa=...`) and checks whether the payee VPA matches the registered merchant profile mapped to the customer\'s current GPS geofence cell.'
        },
        {
          step: 5,
          name: 'Decision & Mitigation Flow',
          description: 'If segmentation detects sticker borders or pHash distance > 8 bits, the payment screen halts with an explicit warning showing the merchant store name mismatch and asks the customer to verify the counter standee.'
        }
      ],
      decision_policy: 'Scores > 0.70 trigger an immediate payment block and notify the merchant via SMS/WhatsApp to inspect their standee. Scores between 0.40 and 0.70 prompt the payer with a bold confirmation showing the store registered name.',
      target_latency: '< 90ms on-device inference, < 30ms API verification'
    },

    data_requirements: {
      streams: [
        'Raw camera viewfinder frame or cropped QR image patch (minimum 512×512 pixels)',
        'Decoded UPI payment payload (Virtual Payment Address, Merchant Category Code, payee name)',
        'Customer mobile device GPS coordinates (lat/long within ±15 meter radius)',
        'Merchant physical premise geofence polygon from onboarding records'
      ],
      feature_store_entities: [
        'merchant_qr_reference_phash (64-bit DCT perceptual hash vector of approved standee)',
        'merchant_registered_vpa_whitelist (authorized settlement handles for the store)',
        'merchant_standee_last_photo_timestamp (record of last physical verification photo)'
      ],
      external_integrations: [
        'On-device Camera SDK filter hook for real-time edge segmentation',
        'Merchant soundbox push notification gateway for immediate physical alert'
      ]
    },

    why_not_implemented: 'Requires an on-device lightweight computer vision model deployed inside payment client SDKs, along with a baseline database of verified physical standee photos captured during merchant ground onboarding. The hackathon MVP focuses on backend tabular transaction telemetry.',
    roadmap_to_production: [
      'Collect 5,000 photos of real-world Indian merchant QR standees (clean, damaged, faded, and deliberately overlay-attacked).',
      'Train and quantize a YOLOv8n-seg model for INT8 execution on Android and iOS neural engines.',
      'Deploy pHash validation in the backend QR resolution API to catch handle mismatches server-side.',
      'Integrate merchant self-verification flow where store owners photograph their QR standee weekly.'
    ]
  },

  sim_swap: {
    id: 'sim_swap',
    title: 'SIM Swap Attack',
    short_name: 'SIM Swap',
    category: 'Authentication & Identity Defense',
    status: 'Concept / Not yet implemented',
    target_metric: 'Recall ≥ 0.95 @ FPR ≤ 0.05% on device reassignment',
    estimated_timeline: '6 weeks engineering + telecom carrier API agreements',
    primary_loss_owner: 'Consumers & Issuing Banks',

    fraud_pattern: {
      summary: 'Attackers obtain a replacement SIM card for a victim\'s mobile number by impersonation or bribing telecom store staff, intercepting SMS OTPs to bind UPI to a fraudulent handset.',
      mechanics: 'The attacker uses leaked KYC documents to file a lost-SIM replacement request with a telecom operator. Once the new SIM activates, the victim\'s phone loses network connectivity. The attacker inserts the SIM into their device, completes UPI SMS device binding, resets the UPI PIN via card number or net-banking credentials, and rapidly drains funds across multiple digital wallets.',
      financial_impact: 'Catastrophic account draining. Losses regularly exceed ₹1,00,000 per incident across linked bank accounts within the first 60 minutes before the victim can reach their telecom operator or bank.',
      why_existing_rules_fail: 'The incoming transaction possesses valid SMS binding and correct UPI PIN entry. Because the phone number matches the bank\'s core records, standard payment switches treat the attacker\'s device as the genuine user until carrier-level IMSI changes are verified.'
    },

    detector_function: {
      summary: 'An event-driven carrier verification detector that queries telecom partner APIs for SIM replacement events within a 72-hour window whenever an account binds to a new hardware fingerprint or initiates high-value transfers.',
      pipeline_steps: [
        {
          step: 1,
          name: 'Device Binding & Hardware Change Detection',
          description: 'Client telemetry records Android ID, hardware serial, and carrier IMSI hash during app startup. Any divergence from the established 90-day device fingerprint triggers an identity audit state.'
        },
        {
          step: 2,
          name: 'Carrier SIM-Swap API Query (GSMA Open Gateway)',
          description: 'Sends an asynchronous check to Airtel, Jio, and Vi carrier gateway endpoints requesting the `sim_swap_timestamp` for the MSISDN.'
        },
        {
          step: 3,
          name: 'Temporal Risk Window Evaluation',
          description: 'Calculates Δt between SIM replacement time and the transaction attempt. If SIM swap occurred < 48 hours prior, high-risk flag `sim_swapped_recent=1` is asserted.'
        },
        {
          step: 4,
          name: 'Velocity & Beneficiary Novelty Correlation',
          description: 'Correlates the carrier status with recipient novelty: transfers to newly added VPA handles or values > ₹5,000 escalate immediately to high severity.'
        },
        {
          step: 5,
          name: 'Cooling-off Enforcement',
          description: 'Automatically places the UPI handle into a mandatory 24-hour transaction cooldown, restricting outbound transfers while allowing inbound credits and notifying the customer via email and push.'
        }
      ],
      decision_policy: 'Confirmed SIM swap < 24 hours ago blocks all outbound transactions immediately. SIM swap between 24 and 72 hours enforces a ₹2,000 per-day velocity ceiling until biometric or in-branch confirmation.',
      target_latency: '< 150ms telecom API lookup (cached with 15-minute TTL)'
    },

    data_requirements: {
      streams: [
        'Mobile phone number (MSISDN) in E.164 standard format',
        'Device hardware ID (IMEI hash, Android ID, Apple IDFV)',
        'Carrier IMSI/ICCID hash reported by mobile banking SDK',
        'Outbound transfer amount and recipient beneficiary age'
      ],
      feature_store_entities: [
        'user_primary_device_history (list of known device hashes with first/last seen dates)',
        'carrier_sim_swap_event_log (timestamp of last SIM reissue from GSMA gateway)',
        'user_daily_outbound_volume_30d (baseline transaction distribution)'
      ],
      external_integrations: [
        'GSMA Open Gateway / Camara SIM Swap API integration (Jio, Airtel, Vodafone Idea)',
        'NPCI Common Library (CL) device binding telemetry hooks'
      ]
    },

    why_not_implemented: 'Requires commercial enterprise agreements and regulatory clearance with Indian telecom operators (Jio, Airtel, Vi) to access real-time SIM-swap notification APIs. The hackathon environment does not have live carrier gateway credentials.',
    roadmap_to_production: [
      'Partner with telecom aggregator (e.g. Tanla / Route Mobile / GSMA Open Gateway) for carrier API access.',
      'Implement an async background polling worker with Redis cache for carrier responses.',
      'Deploy 24-hour cooling-off enforcement in the policy engine for newly bound devices.',
      'Integrate out-of-band email and verified secondary contact alerts for affected users.'
    ]
  },

  account_takeover: {
    id: 'account_takeover',
    title: 'Account Takeover & Credential Stuffing',
    short_name: 'Account Takeover',
    category: 'Authentication & Identity Defense',
    status: 'Concept / Not yet implemented',
    target_metric: 'Precision ≥ 0.94, Recall ≥ 0.88 on automated login surges',
    estimated_timeline: '4 weeks engineering + threat intelligence feed integration',
    primary_loss_owner: 'Merchants & E-commerce Platforms',

    fraud_pattern: {
      summary: 'Botnets execute high-velocity credential stuffing against merchant dashboard logins using credential dumps from third-party data breaches, hijacking merchant portals to alter payout bank details.',
      mechanics: 'Automated distributed bots cycle through compromised email/password combos across hundreds of rotating residential proxy IPs. Once a login succeeds, the attacker enters the merchant settlement settings, swaps the registered payout bank account or settlement VPA to a mule account, and triggers an instant on-demand settlement.',
      financial_impact: 'Entire merchant daily settlement balances (often ₹50,000 to ₹5,00,000+) are siphoned into mule accounts before the genuine business owner discovers access issues.',
      why_existing_rules_fail: 'Once credentials match, basic authentication systems assume identity. If attackers use clean residential proxies matching the merchant\'s home state, simple IP blocklists fail.'
    },

    detector_function: {
      summary: 'A behavioral authentication and sequence anomaly model that tracks typing dynamics, session entropy, impossible travel velocity, and critical setting alteration risk.',
      pipeline_steps: [
        {
          step: 1,
          name: 'Client Environment & Behavioral Fingerprinting',
          description: 'Client script records canvas rendering hashes, WebGL extensions, browser automation flags (Puppeteer/Selenium), and keystroke flight times.'
        },
        {
          step: 2,
          name: 'Breached Credential & Reputation Lookup',
          description: 'Asynchronous check against curated threat intelligence feeds (HaveIBeenPwned API, Spamhaus, internal compromised credential store) on login attempt.'
        },
        {
          step: 3,
          name: 'Impossible Travel & Geovelocity Analysis',
          description: 'Calculates physical velocity required between consecutive logins (e.g. login from Mumbai followed by login from Delhi 12 minutes later implies credential sharing or proxy use).'
        },
        {
          step: 4,
          name: 'Critical Action Anomaly Scoring',
          description: 'Monitors user trajectory within the portal. An immediate navigation to "Bank Accounts / Settlement Settings" following a new-device login elevates the risk score to critical.'
        },
        {
          step: 5,
          name: 'Step-up Authentication & Payout Freeze',
          description: 'Enforces mandatory multi-factor biometric or hardware token verification before allowing any payout bank modification, with a mandatory 48-hour payout lock.'
        }
      ],
      decision_policy: 'High risk (> 0.80) enforces WebAuthn/FIDO2 hardware challenge and freezes instant settlement for 48 hours. Moderate risk (0.50–0.80) requires SMS OTP + email confirmation link.',
      target_latency: '< 50ms evaluation at login API gate'
    },

    data_requirements: {
      streams: [
        'Merchant portal login attempts (email hash, password entropy, user-agent)',
        'Client browser fingerprint (Canvas hash, audio context hash, screen depth)',
        'Source IP address, autonomous system number (ASN), and proxy classification',
        'Post-login event clickstream (sequence of accessed endpoints within 10 minutes)'
      ],
      feature_store_entities: [
        'merchant_known_ip_clusters (frequent login locations over last 180 days)',
        'merchant_payout_account_change_history (timestamps of prior bank detail edits)',
        'merchant_login_failure_count_1h (rolling velocity of bad attempts)'
      ],
      external_integrations: [
        'Threat intelligence feeds for breached credential hash matching',
        'IP reputation and residential proxy detection database (IPQualityScore / MaxMind)'
      ]
    },

    why_not_implemented: 'Requires comprehensive instrumentation of merchant portal session cookies, clickstream logging pipelines, and enterprise threat intelligence subscriptions. The current scope addresses transaction risk rather than portal authentication.',
    roadmap_to_production: [
      'Implement client-side behavioral SDK tracking typing cadence and headless browser markers.',
      'Deploy an automated 48-hour settlement freeze on newly added payout bank accounts.',
      'Integrate k-Anonymity breached credential checking at login submission time.',
      'Set up alerting for impossible travel velocity across successive login events.'
    ]
  },

  voice_auth: {
    id: 'voice_auth',
    title: 'AI Voice Cloning & Vishing Shield',
    short_name: 'Voice Cloning Defense',
    category: 'Audio Forensics & Social Engineering',
    status: 'Concept / Not yet implemented',
    target_metric: 'Equal Error Rate (EER) ≤ 2.0% on ASVspoof-5 benchmark',
    estimated_timeline: '6 weeks engineering + speech ML GPU infrastructure',
    primary_loss_owner: 'High-Value Merchants & Corporate Finance Officers',

    fraud_pattern: {
      summary: 'Attackers use few-shot generative voice synthesis to clone the voices of corporate executives, calling finance personnel to authorize urgent off-cycle wire transfers or override risk blocks.',
      mechanics: 'Attackers harvest 30 seconds of high-clarity voice audio from public webinars, earnings calls, or social media videos of a company executive. Using modern diffusion or autoregressive voice models (e.g. XTTS, ElevenLabs), they generate realistic real-time speech. The attacker calls an accounting desk, spoofing the caller ID, demanding an immediate payment release under the guise of an emergency acquisition.',
      financial_impact: 'Severe enterprise losses. Individual vishing attacks routinely result in losses of ₹10,00,000 to over ₹1,00,00,000 per incident in corporate treasury environments.',
      why_existing_rules_fail: 'Human operators cannot reliably distinguish generative voices over band-limited telephony audio. The request appears internally authenticated, and the transaction is initiated directly by an authorized staff member.'
    },

    detector_function: {
      summary: 'A real-time audio anti-spoofing pipeline that inspects raw speech streams for generative synthetic artifacts, phase discontinuities, and voiceprint cosine similarity during live calls.',
      pipeline_steps: [
        {
          step: 1,
          name: 'RTP Audio Stream Ingestion',
          description: 'The telephony gateway forks live call audio to a WebRTC media engine, converting G.711 / Opus streams into 16kHz uncompressed PCM audio chunks.'
        },
        {
          step: 2,
          name: 'Spectral & Prosodic Feature Extraction',
          description: 'Extracts linear frequency cepstral coefficients (LFCC) and high-order spectrograms to detect the over-smoothed prosody and lack of micro-jitter characteristic of neural voice synthesis.'
        },
        {
          step: 3,
          name: 'Deep Synthetic Voice Classification (AASIST-L)',
          description: 'A pre-trained spectro-temporal graph attention network (AASIST) evaluates the audio chunk for generative vocoder artifacts and neural synthesis signatures.'
        },
        {
          step: 4,
          name: 'Speaker Verification (ECAPA-TDNN x-vector)',
          description: 'Extracts an x-vector embedding of the speaker and calculates cosine similarity against the verified voiceprint enrolled by the legitimate executive.'
        },
        {
          step: 5,
          name: 'Active Challenge Liveness Fallback',
          description: 'If synthetic probability > 0.40, the system triggers an in-band automated prompt requiring the caller to repeat a randomly generated 4-digit code.'
        }
      ],
      decision_policy: 'Synthetic confidence > 0.85 alerts the operator with a flashing "Synthetic Voice Detected" HUD banner and disables one-click payment authorizations for that session.',
      target_latency: '< 450ms decision latency after initial 3 seconds of continuous speech'
    },

    data_requirements: {
      streams: [
        'Live telephony audio stream (16kHz PCM, mono, 200ms frame chunks)',
        'Caller ID signaling metadata (ANI, STIR/SHAKEN attestation level)',
        'Call session parameters (inbound trunk route, telephony codec)',
        'Interactive Voice Response (IVR) DTMF and challenge responses'
      ],
      feature_store_entities: [
        'executive_enrolled_voiceprint_vector (192-dimensional ECAPA-TDNN embedding)',
        'caller_historical_ani_reputation (prior fraud reports associated with phone number)',
        'organization_high_risk_call_logs (past flagged executive impersonations)'
      ],
      external_integrations: [
        'Janus WebRTC or FreeSWITCH media server for real-time RTP audio stream forking',
        'Corporate PBX / SIP trunk gateway integration for live operator alerting'
      ]
    },

    why_not_implemented: 'Requires dedicated audio media streaming servers (WebRTC/SIP), low-latency GPU inference clusters, and enrollment of authorized corporate voiceprints under DPDP Act compliance. The hackathon scope is restricted to payment transaction telemetry.',
    roadmap_to_production: [
      'Set up Janus WebRTC media server integration with test SIP PBX.',
      'Train AASIST-L anti-spoofing model on ASVspoof-5 and Indian English/Hindi voice clone corpora.',
      'Implement an active liveness challenge module using Whisper speech-to-text validation.',
      'Establish DPDP-compliant consent workflows for corporate voiceprint enrollment.'
    ]
  },

  kyc_liveness: {
    id: 'kyc_liveness',
    title: 'Deepfake Video KYC & Injection Sentinel',
    short_name: 'Deepfake KYC Sentinel',
    category: 'Computer Vision & Onboarding Defense',
    status: 'Concept / Not yet implemented',
    target_metric: 'APCER ≤ 1.0%, BPCER ≤ 0.5% (ISO/IEC 30107-3 compliant)',
    estimated_timeline: '8 weeks engineering + GPU cluster + RBI regulatory audit',
    primary_loss_owner: 'Financial Institutions & Regulated Payment Aggregators',

    fraud_pattern: {
      summary: 'Fraud syndicates inject generative face-swap video feeds and virtual camera drivers into mandatory RBI Video Customer Identification Processes (V-CIP) to open synthetic merchant accounts.',
      mechanics: 'Attackers create fake merchant applications using stolen PAN and Aadhaar identity documents. During the live video verification call with an onboarding officer, the attacker uses software such as OBS Virtual Camera or DeepFaceLive to inject real-time generative video puppets mimicking the photo on the stolen ID, responding to basic officer questions.',
      financial_impact: 'Enables creation of untraceable mule merchant accounts. These accounts serve as collection points for fraud rings, laundering hundreds of crores before being shut down by regulators.',
      why_existing_rules_fail: 'Human onboarding officers conducting dozens of verification calls daily suffer fatigue and cannot spot subtle 3D boundary blending or 60Hz screen flicker artifacts on small video windows.'
    },

    detector_function: {
      summary: 'A multi-layer computer vision sentinel operating inside the WebRTC stream to identify virtual camera driver hooks, presentation attack textures, and temporal puppetry artifacts in real time.',
      pipeline_steps: [
        {
          step: 1,
          name: 'Hardware Capture & Driver Interrogation',
          description: 'The onboarding SDK interrogates operating system media device APIs for virtual camera signatures (OBS-Camera, ManyCam, DirectShow software capture filters) and frame timing jitter.'
        },
        {
          step: 2,
          name: 'Face Detection & Normalization (RetinaFace)',
          description: 'Extracts 68 facial landmarks from 5fps video frames, calculating face orientation, bounding box stability, and interpupillary distance.'
        },
        {
          step: 3,
          name: 'Presentation Attack Detection (CDCN++)',
          description: 'Central Difference Convolutional Networks analyze micro-textures to detect screen refresh patterns, printed mask paper grain, and unnatural light reflections.'
        },
        {
          step: 4,
          name: 'Deepfake Temporal Consistency (ViViT)',
          description: 'A Video Vision Transformer assesses temporal coherence across 2-second windows, checking eye blink dynamics, natural pulse variations (remote photoplethysmography / rPPG), and teeth boundary sharpness.'
        },
        {
          step: 5,
          name: 'Interactive Action-Unit Verification',
          description: 'Prompts the user with unpredictable challenges ("Turn head 45° right and blink twice"). Compares action-unit timing against generative rendering lag.'
        }
      ],
      decision_policy: 'Synthetic confidence > 0.70 or virtual camera detection immediately halts automated onboarding and flags the session for supervisory forensic review.',
      target_latency: '< 200ms per frame batch evaluation on GPU inference server'
    },

    data_requirements: {
      streams: [
        'WebRTC uncompressed video stream (minimum 720p @ 15fps)',
        'Client OS device enumeration table (camera driver names, USB vendor IDs)',
        'Browser hardware acceleration and WebGL capability telemetry',
        'Interactive liveness prompt timestamp sequence and user response frames'
      ],
      feature_store_entities: [
        'applicant_identity_document_embedding (FaceNet embedding from submitted PAN card)',
        'device_onboarding_history (prior KYC attempts linked to camera hardware ID)',
        'ip_reputation_kyc_cluster (geo and ASN risk score)'
      ],
      external_integrations: [
        'RBI V-CIP compliant secure video recording repository with 180-day retention',
        'National identity verification API (UIDAI Aadhaar / NSDL PAN verification)'
      ]
    },

    why_not_implemented: 'Requires WebRTC video infrastructure, high-throughput GPU inference nodes, specialized deepfake training datasets, and formal compliance certification under RBI V-CIP guidelines. Out of scope for transaction-level fraud detection.',
    roadmap_to_production: [
      'Build WebRTC video ingestion service with frame sampling pipeline.',
      'Train CDCN++ and ViViT models on diverse Indian demographic face datasets and generative swap tools.',
      'Implement OS-level driver integrity checks in the web/mobile onboarding SDK.',
      'Conduct independent third-party ISO/IEC 30107-3 presentation attack detection audit.'
    ]
  },

  chargeback_responder: {
    id: 'chargeback_responder',
    title: 'Automated Chargeback Evidence Responder',
    short_name: 'Chargeback Responder',
    category: 'Dispute Automation & Post-Transaction',
    status: 'Concept / Not yet implemented',
    target_metric: 'F1 ≥ 0.88 on dispute win/loss labels, +15pp win-rate lift',
    estimated_timeline: '5 weeks engineering + courier & card network API integrations',
    primary_loss_owner: 'Online Merchants & E-commerce Retailers',

    fraud_pattern: {
      summary: 'Consumers exploit card network dispute mechanisms to falsely claim non-receipt or unauthorized use ("friendly fraud") after receiving high-value orders, forcing merchant chargeback debits.',
      mechanics: 'A customer orders electronics or luxury goods, takes delivery at their doorstep, and signs the delivery receipt. Three weeks later, the customer files a chargeback with their card issuer claiming "Goods not received" or "Card stolen / unauthorized transaction". If the merchant fails to respond within the 7-day dispute window with formatted proof, the merchant loses both the goods and the funds.',
      financial_impact: 'Significant merchant revenue leakage. Average chargeback cost includes the transaction amount, ₹1,500 bank penalty fee, and 45 minutes of manual back-office document gathering per dispute.',
      why_existing_rules_fail: 'Preventive rules only inspect the checkout phase. Friendly fraud occurs 15 to 90 days after legitimate authorization by the authentic cardholder.'
    },

    detector_function: {
      summary: 'An automated evidence collector and win-probability model that correlates courier delivery records, IP geolocation, 3DS authentication proofs, and customer chat logs into compliant representment packages.',
      pipeline_steps: [
        {
          step: 1,
          name: 'Dispute Webhook Ingestion',
          description: 'Payment gateway dispute notification initiates an asynchronous evidence bundling job with a 5-minute internal SLA.'
        },
        {
          step: 2,
          name: 'Multi-Source Evidence Harvesting',
          description: 'Queries courier logistics APIs (Delhivery, BlueDart) for delivery GPS coordinates, doorstep photos, and recipient signatures. Pulls 3DS callback logs and customer support chat transcripts.'
        },
        {
          step: 3,
          name: 'Text & Visual Embedding Extraction',
          description: 'Runs sentence-transformers on support chat and email logs to extract customer admissions of receipt. Normalizes delivery proof images for legibility.'
        },
        {
          step: 4,
          name: 'Win Probability Scoring (LightGBM + STR Estimator)',
          description: 'A 200-feature model predicts the merchant\'s probability of prevailing in arbitration, applying a Selective Transition Residual (STR) estimator to correct for issuer reporting bias.'
        },
        {
          step: 5,
          name: 'Representment Package Generation & Auto-Submit',
          description: 'Formats evidence into standard Visa VROL / Mastercard MasterCom PDF bundles. If win probability > 0.92, auto-submits directly to the payment gateway; otherwise routes to an analyst queue.'
        }
      ],
      decision_policy: 'Scores > 0.92 trigger immediate auto-submission. Scores 0.40–0.92 generate a ranked evidence dossier for one-click human review. Scores < 0.40 accept liability to save arbitration fees.',
      target_latency: '< 2 minutes end-to-end evidence assembly'
    },

    data_requirements: {
      streams: [
        'Dispute initiation webhook payload (Dispute ID, Reason Code, Disputed Amount, Due Date)',
        'Payment gateway callback record (3DS challenge status, CAVV/ECI indicators, UPI UTR)',
        'Logistics partner API response (Delivery timestamp, GPS coordinates, signature bitmap, POD photo)',
        'Customer communication history (Zendesk / WhatsApp customer support transcripts)'
      ],
      feature_store_entities: [
        'customer_historical_dispute_rate (past chargeback volume across merchant network)',
        'issuing_bank_dispute_leniency_score (historical win rate segmented by card issuer)',
        'product_sku_dispute_propensity (known dispute rates for specific high-value items)'
      ],
      external_integrations: [
        'Courier logistics tracking webhooks (Delhivery, Ecom Express, Bluedart)',
        'Payment gateway dispute representment APIs (Razorpay / PayU / Cashfree dispute endpoints)'
      ]
    },

    why_not_implemented: 'Requires deep production API integrations with external courier systems, access to historical dispute win/loss adjudication labels, and compliance formatting for card brand representment. The hackathon MVP is centered on real-time transaction scoring.',
    roadmap_to_production: [
      'Build webhook listeners for payment gateway dispute life-cycle events.',
      'Integrate courier logistics APIs for automated retrieval of digital proof-of-delivery (POD).',
      'Train LightGBM representment scoring model using historical win/loss data with STR correction.',
      'Implement auto-submission connector adhering to card brand PDF evidence formatting specifications.'
    ]
  },

  return_risk_scorer: {
    id: 'return_risk_scorer',
    title: 'AI-Doctored Return Risk Scorer',
    short_name: 'Return Fraud Scorer',
    category: 'Dispute Automation & Post-Transaction',
    status: 'Concept / Not yet implemented',
    target_metric: 'PR-AUC ≥ 0.90 on synthetic damage photo forensics',
    estimated_timeline: '5 weeks engineering + computer vision forensic dataset',
    primary_loss_owner: 'E-commerce & Direct-to-Consumer Merchants',

    fraud_pattern: {
      summary: 'Customers use consumer generative AI tools (Midjourney, Stable Diffusion, Photoshop Generative Fill) to fabricate realistic damage photos on newly delivered items to claim instant refunds without returning products.',
      mechanics: 'The buyer purchases an expensive item (designer apparel, smartphone, luxury watch). Upon delivery, they take a clean photo of the product, use generative AI tools to add realistic screen cracks, fabric tears, or water damage, and submit a return claim. Merchants with customer-friendly auto-refund policies credit the money without requiring a physical return or doorstep inspection.',
      financial_impact: 'Severe direct inventory and refund loss for e-commerce retailers, often costing 2–5% of total merchant gross merchandise value (GMV) in affected apparel and electronics categories.',
      why_existing_rules_fail: 'Return portals rely entirely on image uploads and customer claim text. Conventional rule systems cannot inspect pixel-level generative artifacts or detect when an image has been manipulated.'
    },

    detector_function: {
      summary: 'A two-stage forensic pipeline combining convolutional generative-artifact detection with customer return history to distinguish real physical damage from AI synthesis.',
      pipeline_steps: [
        {
          step: 1,
          name: 'Image Preprocessing & Metadata Verification',
          description: 'Extracts EXIF metadata (lens model, camera software, creation timestamp) and checks for discrepancies against the customer\'s registered device and claim timestamp.'
        },
        {
          step: 2,
          name: 'Generative Artifact Detection (EfficientNet-B3)',
          description: 'Analyzes high-frequency Discrete Cosine Transform (DCT) residuals and Steganalysis Rich Model (SRM) noise patterns to identify generative diffusion model footprints.'
        },
        {
          step: 3,
          name: 'Text-Image Claim Alignment (CLIP)',
          description: 'Compares the customer\'s textual claim (e.g. "shattered display") against fine-tuned CLIP visual embeddings to verify whether the damage visual matches the reported failure mode.'
        },
        {
          step: 4,
          name: 'Customer Return History Scoring',
          description: 'LightGBM model assesses the customer\'s historical return velocity, return-to-order ratio, and shared shipping address graph linkages with known refund abuse rings.'
        },
        {
          step: 5,
          name: 'Calibrated Policy Routing',
          description: 'Fuses forensic score and customer risk score to assign an automated action (Auto-approve, Require 360° uncut video, or Enforce physical courier inspection).'
        }
      ],
      decision_policy: 'Scores < 0.20 auto-approve refund. Scores 0.20–0.60 require a 360° unedited video upload. Scores > 0.60 require physical return with courier doorstep validation before refund issuance.',
      target_latency: '< 800ms per image forensic inspection'
    },

    data_requirements: {
      streams: [
        'High-resolution return claim photos submitted by the customer',
        'Customer return claim text and requested refund method',
        'EXIF camera metadata and file creation attributes',
        'Order and delivery fulfillment timestamps'
      ],
      feature_store_entities: [
        'customer_return_rate_90d (ratio of returned items to total orders)',
        'customer_refund_value_ratio (total refunded amount vs lifetime spend)',
        'shipping_address_entity_graph (graph clustering shared addresses and phone numbers)'
      ],
      external_integrations: [
        'Merchant e-commerce return portal webhook and image CDN',
        'Reverse logistics courier inspection API for doorstep verification'
      ]
    },

    why_not_implemented: 'Requires specialized training on generative AI synthetic image datasets, integration with return portal media uploads, and reverse logistics courier workflows. Deferred in favor of UPI payment risk in the MVP.',
    roadmap_to_production: [
      'Curate a benchmark dataset of real versus AI-manipulated product damage photos.',
      'Train EfficientNet-B3 forensic classifier with multi-stage compression augmentations.',
      'Implement video evidence upload and automated verification workflows in the return portal.',
      'Deploy graph entity resolution to detect coordinated return syndicates sharing mule addresses.'
    ]
  },

  review_ring_detector: {
    id: 'review_ring_detector',
    title: 'Coordinated Review Ring & Bot Sentinel',
    short_name: 'Abuse-Ring Sentinel',
    category: 'Graph ML & Market Integrity',
    status: 'Concept / Not yet implemented',
    target_metric: 'Cluster F1 ≥ 0.85 on synthetic and investigated ring labels',
    estimated_timeline: '4 weeks engineering + graph streaming infrastructure',
    primary_loss_owner: 'Marketplaces, Honest Merchants, & Consumers',

    fraud_pattern: {
      summary: 'Syndicates of fake reviewer accounts coordinate synchronized 5-star rating surges for fraudulent merchants or coordinated 1-star review bombings against competing stores.',
      mechanics: 'Abuse rings operate hundreds of bot accounts or recruited gig workers through Telegram groups. On command, they publish batches of reviews within tight time windows, cycling through residential proxies and using templated or LLM-generated praise to manipulate merchant trust scores and marketplace search algorithms.',
      financial_impact: 'Distorts marketplace trust. Legitimate merchants lose customer conversion to deceptive sellers who inflate ratings, while consumers suffer fraud by buying poor-quality or counterfeit items.',
      why_existing_rules_fail: 'Individual reviews appear benign and grammatically correct. Detection is impossible on a single-review level; the threat only becomes visible when analyzing coordinated patterns across bipartite user-product graphs.'
    },

    detector_function: {
      summary: 'A graph neural network and behavioral entropy engine that detects synchronized reviewer communities, IP subnet co-occurrences, and linguistic LLM perplexity patterns.',
      pipeline_steps: [
        {
          step: 1,
          name: 'Review Telemetry & Behavioral Ingestion',
          description: 'Captures review submission metadata along with client-side behavioral SDK signals: typing velocity, paste events, focus duration, and scroll depth.'
        },
        {
          step: 2,
          name: 'Linguistic & Perplexity Scoring',
          description: 'Runs a lightweight language model (LLaMA-3-8B / RoBERTa) to calculate text perplexity, burstiness, and stylometric n-gram overlap against known review templates.'
        },
        {
          step: 3,
          name: 'Bipartite Interaction Graph Construction',
          description: 'Constructs an evolving bipartite graph linking user accounts to reviewed products, with user-user edges formed by shared device hashes, IP subnets, and temporal proximity.'
        },
        {
          step: 4,
          name: 'Community Detection & Ring Scoring (Leiden + GraphSAGE)',
          description: 'The Leiden algorithm identifies tightly clustered reviewer sub-graphs. A GraphSAGE neural network scores the probability that the cluster represents an artificial ring.'
        },
        {
          step: 5,
          name: 'Automated Market Actioning',
          description: 'Accounts scored in confirmed rings are shadow-banned (reviews visible only to the author) and review velocity limits are imposed on the recipient merchant profile.'
        }
      ],
      decision_policy: 'Cluster ring probability > 0.80 shadow-bans all cluster reviews and flags the merchant for review audit. Cluster score 0.50–0.80 strips verified buyer badges and limits review weight.',
      target_latency: '< 5 seconds asynchronous graph update after review submission'
    },

    data_requirements: {
      streams: [
        'Review submission text, star rating, and timestamp',
        'Reviewer user identifier, account creation date, and purchase verification flag',
        'Client SDK telemetry (keystroke timing, paste events, session duration)',
        'Network telemetry (client IP, ASN, subnet, VPN/Tor indicator)'
      ],
      feature_store_entities: [
        'user_review_velocity_24h (number of reviews submitted in 24 hours)',
        'reviewer_product_category_diversity (entropy of reviewed categories)',
        'merchant_review_burst_score (Hawkes process coordination metric)'
      ],
      external_integrations: [
        'Marketplace review submission event bus (Kafka topic `market.reviews`)',
        'Order management system API for verified purchase validation'
      ]
    },

    why_not_implemented: 'Requires full review submission event pipelines, streaming graph database infrastructure (e.g. Neo4j / Memgraph / AWS Neptune), and marketplace order integration. Outside the scope of UPI transaction defense.',
    roadmap_to_production: [
      'Instrument review submission forms with client-side behavioral telemetry.',
      'Build Kafka-to-graph streaming ingestion pipeline with automated edge weighting.',
      'Train GraphSAGE model on synthetic review ring topologies and historical audit cases.',
      'Deploy shadow-ban enforcement and review weighting APIs to the marketplace frontend.'
    ]
  }
};

// Export for module and browser global contexts
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { FRAUD_CONCEPTS };
}
