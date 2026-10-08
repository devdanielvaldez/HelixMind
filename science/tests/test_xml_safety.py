from __future__ import annotations

import unittest

from medai_science.sources.clinvar import ClinVarClient
from medai_science.sources.dailymed_labels import DailyMedLabelClient
from medai_science.sources.sifts import _parse_sifts_xml
from medai_science.sources.xml_safety import reject_xml_declarations


def _utf16_entity_document(root: str) -> bytes:
    return (
        '<?xml version="1.0" encoding="UTF-16"?>'
        '<!DOCTYPE root [<!ENTITY payload "expanded">]>'
        f'<{root}>&payload;</{root}>'
    ).encode("utf-16")


class XmlSafetyTests(unittest.TestCase):
    def test_rejects_utf16_entity_expansion_across_source_parsers(self) -> None:
        payloads = (
            ("SIFTS", lambda: _parse_sifts_xml(_utf16_entity_document("entry"), "1abc", "P05067")),
            ("ClinVar", lambda: ClinVarClient._parse_vcv_locations(_utf16_entity_document("VariationArchiveSet"))),
            ("DailyMed", lambda: DailyMedLabelClient._interaction_section(_utf16_entity_document("document"))),
        )
        for name, parse in payloads:
            with self.subTest(source=name), self.assertRaisesRegex(ValueError, "UTF-8"):
                parse()

    def test_rejects_declarations_with_spacing_and_malformed_utf8(self) -> None:
        with self.assertRaisesRegex(ValueError, "declarations"):
            reject_xml_declarations(b"<root/><!--safe--><! DOCTYPE root>", "test")
        with self.assertRaisesRegex(ValueError, "UTF-8"):
            reject_xml_declarations(b"<root>\xff</root>", "test")

    def test_accepts_utf8_bom_and_regular_xml(self) -> None:
        reject_xml_declarations(b"\xef\xbb\xbf<root />", "test")


if __name__ == "__main__":
    unittest.main()
