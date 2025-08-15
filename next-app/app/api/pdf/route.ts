import { NextRequest, NextResponse } from "next/server";
import { PDFDocument } from "pdf-lib"
import sharp from "sharp";


export default async function POST (req: NextRequest) {
    try {
        const formData = await req.formData()
        const pdfFile = formData.get("file") as File
        if(!pdfFile){
            return NextResponse.json({
                message: "Attach a file"
            }, { status: 411 })
        }    
        
        const extension = pdfFile.name.split(".").pop()?.toLowerCase()
        if(extension !== "pdf") {
            return NextResponse.json({
                message: "Please attach valid pdf file"
            }, { status: 400 })
        }

        const arrayBuffer = await pdfFile.arrayBuffer()
        const pdfDoc = await PDFDocument.load(arrayBuffer)

        const ocrResults: any[] = []

        for (let i = 0; i < pdfDoc.getPageCount(); i++) {
            const singlePagePdf = await PDFDocument.create()
            const [copiedPage] = await singlePagePdf.copyPages(pdfDoc, [i])
            singlePagePdf.addPage(copiedPage)
            const singlePagePdfBytes = await singlePagePdf.save()

            const imageBuffer = await sharp(Buffer.from(singlePagePdfBytes))
                .webp()
                .toBuffer()

            const forwardFormData = new FormData()
            forwardFormData.append("file", new Blob([imageBuffer], { type: "image/webp" }), `page-${i+1}.webp`); 

            const fastApiResponse = await fetch(process.env.FAST_API_LINK as string, {
                method: "POST",
                body: forwardFormData
            })

            const pageResult = await fastApiResponse.json()
            ocrResults.push({ page: i + 1, result: pageResult })

            return NextResponse.json({
                data: ocrResults
            })
        }
    } catch (error) {
        console.error(error)
        return NextResponse.json({
            message: "Server Error"
        }, { status: 500 })
    }
}