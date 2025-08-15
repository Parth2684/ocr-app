import { NextRequest, NextResponse } from "next/server";



export default async function POST (req: NextRequest) {
    try {
        const formData = await req.formData()
        const file = formData.get("file") as File;
        if(!file) {
            return NextResponse.json({
                message: "No file attached"
            }, { status: 411 })
        }

        const forwardFormData = new FormData()
        forwardFormData.append("file", file, file.name)

        const fastApiResponse = await fetch(process.env.FAST_API_LINK as string, {
            method: "POST",
            body: forwardFormData
        })

        const data = await fastApiResponse.json()

        return NextResponse.json(data)
    } catch (error) {
        console.error(error)
        return NextResponse.json({
            message: "Server Error"
        }, { status: 500 })
    }
}