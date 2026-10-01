package com.example.shipping;

import java.math.BigDecimal;
import java.time.LocalDate;
import java.util.List;

public class CustomsDeclaration {

    private final String declarationNumber;
    private final LocalDate exportDate;
    private final List<LineItem> lineItems;

    public CustomsDeclaration(String declarationNumber, LocalDate exportDate, List<LineItem> lineItems) {
        this.declarationNumber = declarationNumber;
        this.exportDate = exportDate;
        this.lineItems = lineItems;
    }

    /** Value for duty: invoice value plus freight and insurance, excluding tax. */
    public BigDecimal customsValue(BigDecimal freight, BigDecimal insurance) {
        BigDecimal invoice = lineItems.stream()
            .map(LineItem::declaredValue)
            .reduce(BigDecimal.ZERO, BigDecimal::add);
        return invoice.add(freight).add(insurance);
    }

    public boolean requiresLicence(String countryOfDestination) {
        return lineItems.stream().anyMatch(item -> item.isControlled() && item.origin() != countryOfDestination);
    }

    public int lineCount() {
        return lineItems.size();
    }
}
