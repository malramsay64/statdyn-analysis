import {XYGlyph, XYGlyphView, XYGlyphData} from "models/glyphs/xy_glyph"
import {DistanceSpec, AngleSpec} from "core/vectorization"
import {LineMixinVector, FillMixinVector} from "core/property_mixins"
import {Line, Fill} from "core/visuals"
import {Arrayable} from "core/types"
import * as p from "core/properties"
import {IBBox} from "core/util/bbox"
import {Rect} from "core/util/spatial"
import {Context2d} from "core/util/canvas"

export interface TrimerData extends XYGlyphData {
    _angle: Arrayable<number>

    distance: number
    radius: number
    scale: number
    ang: number

    size2: number
}

export interface TrimerView extends TrimerData {}

export class TrimerView extends XYGlyphView {
    model: Trimer
    visuals: Trimer.Visuals

    protected _set_data(): void {
        this.size2 = 1.3
        this.distance = 1.0
        this.radius = 0.6
        this.ang = Math.PI/3
    }

    protected _map_data(): void {
        // Calculate the scaling factor of the figure
        this.scale = this.sdist(this.renderer.xscale, this._x, [1], 'center')[0]
    }

    protected _render(ctx: Context2d, indices: number[], {sx, sy, scale}: TrimerData): void {
        const sradius: number = scale * this.radius
        const sdist: number = scale * this.distance
        for (const i of indices) {
            if (isNaN(sx[i] + sy[i] + this._angle[i]))
                continue

            ctx.beginPath()
            ctx.arc(sx[i], sy[i], scale, 0, 2 * Math.PI)
            if (this.visuals.fill.doit) {
                this.visuals.fill.set_vectorize(ctx, i)
                ctx.fill()
            }
            if (this.visuals.line.doit) {
                this.visuals.line.set_vectorize(ctx, i)
            }
            ctx.stroke()

            ctx.beginPath()
            ctx.ellipse(
                sx[i] + sdist * Math.sin(this._angle[i] - this.ang),
                sy[i] + sdist * Math.cos(this._angle[i] - this.ang),
                sradius,
                sradius,
                Math.PI-this.ang/2-this._angle[i],
                -2.0 , 2.0
            )
            if (this.visuals.fill.doit) {
                this.visuals.fill.set_vectorize(ctx, i)
                ctx.fill()
            }
            if (this.visuals.line.doit) {
                this.visuals.line.set_vectorize(ctx, i)
            }
            ctx.stroke()

            ctx.beginPath()
            ctx.ellipse(
                sx[i] + sdist * Math.sin(this._angle[i] + this.ang),
                sy[i] + sdist * Math.cos(this._angle[i] + this.ang),
                sradius,
                sradius,
                this.ang/2-this._angle[i],
                -2.0 , 2.0
            )
            if (this.visuals.fill.doit) {
                this.visuals.fill.set_vectorize(ctx, i)
                ctx.fill()
            }
            if (this.visuals.line.doit) {
                this.visuals.line.set_vectorize(ctx, i)
            }
            ctx.stroke()
        }
    }

    draw_legend_for_index(ctx: Context2d, {x0, y0, x1, y1}: IBBox, index: number): void {
        const len = index + 1

        const sx: number[] = new Array(len)
        sx[index] = (x0 + x1)/2
        const sy: number[] = new Array(len)
        sy[index] = (y0 + y1)/2

        const d = Math.min(Math.abs(x1 - x0), Math.abs(y1 - y0))*0.8

        const scale: number = d

        this._render(ctx, [index], {sx, sy, scale} as any) // XXX
    }

    protected _bounds({minX, maxX, minY, maxY}: Rect): Rect {
        return {
            minX: minX - this.size2,
            maxX: maxX + this.size2,
            minY: minY - this.size2,
            maxY: maxY + this.size2,
        }
    }
}

export namespace Trimer {
    export interface Mixins extends LineMixinVector, FillMixinVector {}

    export interface Attrs extends XYGlyph.Attrs, Mixins {
        angle: AngleSpec
    }

    export interface Props extends XYGlyph.Props {
        angle: p.AngleSpec
    }

    export interface Visuals extends XYGlyph.Visuals {
        line: Line
        fill: Fill
    }
}

export interface Trimer extends Trimer.Attrs {}

export class Trimer extends XYGlyph {

    properties: Trimer.Props

    constructor(attrs?: Partial<Trimer.Attrs>) {
        super(attrs)
    }

    static initClass(): void {
        this.prototype.type = 'Trimer'
        this.prototype.default_view = TrimerView

        this.mixins(['line', 'fill'])
        this.define({
            angle:  [ p.AngleSpec,   0.0 ],
        })
    }
}
Trimer.initClass()
